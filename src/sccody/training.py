"""Frozen-representation population transport with bounded cell batches."""
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import logging
import numpy as np
import pandas as pd
import torch
from geomloss import SamplesLoss
from torch.nn import functional as F
from .data import expression_rows

logger = logging.getLogger(__name__)


@dataclass
class Population:
    time: float
    stratum: tuple[str, ...]
    cell_type: str | None
    rows: np.ndarray
    latent: torch.Tensor
    codes: torch.Tensor

    @property
    def family(self):
        return self.stratum, self.cell_type


def tensor_rows(adata, pipeline, rows):
    return torch.as_tensor(expression_rows(adata, pipeline.config, rows, pipeline.gene_indices, pipeline.library_indices), device=pipeline.device)


def decode(pipeline, latent):
    return F.softplus(pipeline.representation.decode(latent))


def distance(pipeline):
    t = pipeline.config.training
    # Tensorized backend bounds memory to configured cell batches on CPU and GPU.
    return SamplesLoss("sinkhorn", p=2, blur=t.sinkhorn_blur, scaling=t.sinkhorn_scaling, backend="tensorized")


def flow(pipeline, source_time, target_time, latent, codes):
    s = (float(source_time) - pipeline.time_origin) / pipeline.time_scale
    t = (float(target_time) - pipeline.time_origin) / pipeline.time_scale
    if pipeline.config.model.scope == "cell_type":
        return pipeline.transport(s, t, latent, codes)
    return pipeline.transport(s, t, latent)


def populations(adata, pipeline, rows, *, unit=None):
    d, t = pipeline.config.data, pipeline.config.training
    unit = t.trajectory_unit if unit is None else unit
    metadata = adata.obs.iloc[rows].copy()
    metadata["_time"] = pd.to_numeric(metadata[d.time_key]).astype(float)
    metadata["_sample"] = metadata[d.sample_key].astype(str)
    metadata["_row"] = rows
    keys = ["_sample" if unit == "sample" else "_time", *d.stratify_keys]
    if pipeline.config.model.scope == "cell_type":
        metadata[d.cell_type_key] = metadata[d.cell_type_key].astype(str)
        keys.append(d.cell_type_key)
    nodes = []
    pipeline.representation.eval()
    with torch.no_grad():
        for _, group in metadata.groupby(keys, observed=True, sort=True):
            group_rows = group["_row"].to_numpy(dtype=int)
            chunks = []
            for start in range(0, len(group_rows), t.representation_batch_size):
                values = tensor_rows(adata, pipeline, group_rows[start:start + t.representation_batch_size])
                chunks.append(pipeline.representation.encode(values)[0].cpu())
            ct = str(group[d.cell_type_key].iloc[0]) if pipeline.config.model.scope == "cell_type" else None
            if ct is not None and ct not in pipeline.cell_types:
                raise ValueError(f"Cell type {ct!r} was not present in training")
            code = pipeline.cell_types.index(ct) if ct is not None else 0
            nodes.append(Population(float(group["_time"].iloc[0]),
                                    tuple(str(group[k].iloc[0]) for k in d.stratify_keys), ct,
                                    group_rows, torch.cat(chunks), torch.full((len(group_rows),), code, dtype=torch.long)))
    return nodes


def train_representation(adata, pipeline, rows, outdir):
    t = pipeline.config.training
    rng = np.random.default_rng(t.seed)
    optimizer = torch.optim.AdamW(pipeline.representation.parameters(), lr=t.representation_lr, weight_decay=1e-4, betas=(0.9, 0.99))
    history = []
    for epoch in range(1, t.representation_epochs + 1):
        pipeline.representation.train()
        shuffled = rng.permutation(rows)
        total = 0.0
        for start in range(0, len(rows), t.representation_batch_size):
            values = tensor_rows(adata, pipeline, shuffled[start:start + t.representation_batch_size])
            optimizer.zero_grad(set_to_none=True)
            mean, logvar = pipeline.representation.encode(values)
            latent = pipeline.representation.reparameterize(mean, logvar)
            raw = pipeline.representation.decode(latent)
            recon = raw if pipeline.config.model.representation.endswith("_nb") else F.softplus(raw)
            loss = pipeline.representation.loss(values, recon, mean, logvar, beta_kl=t.beta)[0]
            if not torch.isfinite(loss):
                raise RuntimeError("Representation loss became nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pipeline.representation.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(values)
        history.append({"epoch": epoch, "loss": total / len(rows)})
        logger.info("Representation epoch %d/%d: %.6g", epoch, t.representation_epochs, history[-1]["loss"])
        pd.DataFrame(history).to_csv(outdir / "representation_history.csv", index=False)
    pipeline.representation.eval()
    pipeline.representation.requires_grad_(False)


def sample_node(node, size, rng, device):
    indices = rng.choice(len(node.rows), min(size, len(node.rows)), replace=False)
    return node.latent[indices].to(device), node.codes[indices].to(device), node.rows[indices]


def validation_score(adata, pipeline, training_nodes, validation_nodes):
    """Known-time expression score on sample-disjoint validation populations."""
    rng = np.random.default_rng(pipeline.config.training.seed + 1)
    sinkhorn = distance(pipeline)
    scores = []
    with torch.no_grad():
        for target in validation_nodes:
            sources = [n for n in training_nodes if n.family == target.family and n.time != target.time]
            if not sources:
                raise ValueError(f"Validation population needs a matching training source at a different time: {target.family}")
            source = min(sources, key=lambda n: (abs(n.time - target.time), n.time))
            z, codes, _ = sample_node(source, pipeline.config.training.cell_batch_size, rng, pipeline.device)
            _, _, target_rows = sample_node(target, pipeline.config.training.cell_batch_size, rng, pipeline.device)
            predicted = decode(pipeline, flow(pipeline, source.time, target.time, z, codes))
            scores.append(float(sinkhorn(predicted, tensor_rows(adata, pipeline, target_rows)).detach()))
    if not scores or not np.isfinite(scores).all():
        raise ValueError("Validation selection requires finite supported validation populations")
    # One population per sample (or sample/cell-type), so large cell counts do not set weights.
    return float(np.mean(scores))


def train_transport(adata, pipeline, nodes, validation_nodes, outdir):
    t = pipeline.config.training
    rng = np.random.default_rng(t.seed)
    families = defaultdict(list)
    for node in nodes:
        families[node.family].append(node)
    for family, members in families.items():
        if len({n.time for n in members}) < 2:
            raise ValueError(f"Training stratum/cell type {family} needs at least two times")
    groups = list(families.values())
    optimizer = torch.optim.Adam(pipeline.transport.parameters(), lr=t.flow_lr)
    sinkhorn = distance(pipeline)
    best_score, best_state, best_epoch = float("inf"), None, None
    history = []
    # Validate support before starting the production training budget.
    if t.selection == "validation_gene_ot":
        validation_score(adata, pipeline, nodes, validation_nodes)
    for epoch in range(1, t.flow_epochs + 1):
        pipeline.transport.train()
        totals = []
        for _ in range(t.steps_per_epoch):
            members = groups[int(rng.integers(len(groups)))]
            # Choose two distinct times, then a sample at each time; support reverse transport.
            times = sorted({n.time for n in members})
            chosen = rng.choice(times, 2, replace=False)
            s, target_time = chosen if t.bidirectional else sorted(chosen)
            sources = [n for n in members if n.time == s]
            targets = [n for n in members if n.time == target_time]
            source = sources[int(rng.integers(len(sources)))]
            target = targets[int(rng.integers(len(targets)))]
            z, codes, _ = sample_node(source, t.cell_batch_size, rng, pipeline.device)
            z_target, _, rows_target = sample_node(target, t.cell_batch_size, rng, pipeline.device)
            optimizer.zero_grad(set_to_none=True)
            predicted = flow(pipeline, s, target_time, z, codes)
            loss = t.latent_ot_weight * sinkhorn(predicted, z_target)
            if t.gene_ot_weight:
                loss = loss + t.gene_ot_weight * sinkhorn(decode(pipeline, predicted), tensor_rows(adata, pipeline, rows_target))
            # Endpoint consistency compares independent source populations at a common target.
            if t.endpoint_weight:
                other = [n for n in members if n is not source and n is not target and n.time != target_time]
                if other:
                    second = other[int(rng.integers(len(other)))]
                    z2, c2, _ = sample_node(second, t.cell_batch_size, rng, pipeline.device)
                    endpoint = flow(pipeline, second.time, target_time, z2, c2)
                    loss = loss + t.endpoint_weight * sinkhorn(predicted, endpoint)
            if not torch.isfinite(loss):
                raise RuntimeError("Transport loss became nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pipeline.transport.parameters(), 5.0)
            optimizer.step()
            totals.append(float(loss.detach()))
        pipeline.transport.eval()
        score = validation_score(adata, pipeline, nodes, validation_nodes) if t.selection == "validation_gene_ot" else None
        if t.selection == "last" or score < best_score:
            best_score = score if score is not None else float("inf")
            best_state = deepcopy(pipeline.transport.state_dict())
            best_epoch = epoch
        history.append({"epoch": epoch, "train_loss": float(np.mean(totals)), "validation_gene_ot": score})
        pd.DataFrame(history).to_csv(outdir / "flow_history.csv", index=False)
        logger.info("Transport epoch %d/%d: loss=%.6g validation=%s", epoch, t.flow_epochs, history[-1]["train_loss"], score)
    pipeline.transport.load_state_dict(best_state)
    pipeline.transport.eval()
    pipeline.selection = {"criterion": t.selection, "epoch": best_epoch,
                          "validation_score": None if t.selection == "last" else best_score,
                          "test_samples_used": False}
    return history
