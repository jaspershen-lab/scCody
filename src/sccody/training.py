"""Frozen-representation population transport with bounded cell batches."""
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import logging
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from .data import expression_rows
from .losses import sinkhorn_divergence
from .trajectories import training_pairs, endpoint_triples

logger = logging.getLogger(__name__)


@dataclass
class Population:
    time: float
    stratum: tuple[str, ...]
    cell_type: str | None
    rows: np.ndarray
    latent: torch.Tensor
    codes: torch.Tensor
    sample_id: str

    @property
    def family(self):
        return self.stratum, self.cell_type


def tensor_rows(adata, pipeline, rows):
    return torch.as_tensor(expression_rows(adata, pipeline.config, rows, pipeline.gene_indices, pipeline.library_indices), device=pipeline.device)


def decode(pipeline, latent):
    raw = pipeline.representation.decode(latent)
    return raw if pipeline.config.model.no_softplus_decode else F.softplus(raw)


def distance(pipeline):
    t = pipeline.config.training
    return lambda source, target: sinkhorn_divergence(
        source, target, blur=t.sinkhorn_blur, scaling=t.sinkhorn_scaling)


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
            code = pipeline.cell_types.index(ct) if ct is not None else 0
            nodes.append(Population(float(group["_time"].iloc[0]),
                                    tuple(str(group[k].iloc[0]) for k in d.stratify_keys), ct,
                                    group_rows, torch.cat(chunks), torch.full((len(group_rows),), code, dtype=torch.long),
                                    str(group["_sample"].iloc[0])))
    return nodes


def train_representation(adata, pipeline, rows):
    t = pipeline.config.training
    rng = np.random.default_rng(t.seed)
    optimizer = torch.optim.AdamW(pipeline.representation.parameters(), lr=t.representation_lr, weight_decay=1e-4, betas=(0.9, 0.99))
    for epoch in range(1, t.representation_epochs + 1):
        pipeline.representation.train()
        shuffled = rng.permutation(rows)
        total = 0.0
        for start in range(0, len(rows), t.representation_batch_size):
            values = tensor_rows(adata, pipeline, shuffled[start:start + t.representation_batch_size])
            optimizer.zero_grad(set_to_none=True)
            mean, logvar = pipeline.representation.encode(values)
            latent = pipeline.representation.reparameterize(mean, logvar)
            recon = decode(pipeline, latent)
            loss = pipeline.representation.loss(values, recon, mean, logvar, beta_kl=t.beta)[0]
            if not torch.isfinite(loss):
                raise RuntimeError("Representation loss became nonfinite")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(values)
        logger.info("Representation epoch %d/%d: %.6g", epoch, t.representation_epochs, total / len(rows))
    pipeline.representation.eval()
    pipeline.representation.requires_grad_(False)


def sample_node(node, size, rng, device):
    indices = rng.choice(len(node.rows), min(size, len(node.rows)), replace=False)
    return node.latent[indices].to(device), node.codes[indices].to(device), node.rows[indices]


def trajectory_step(adata, pipeline, members, rng, optimizer, sinkhorn):
    """Average population-pair losses while releasing each graph after backward."""
    t = pipeline.config.training
    members = sorted(members, key=lambda n: (n.time, n.sample_id))
    if len(members) > t.trajectory_nodes:
        positions = rng.choice(len(members), size=t.trajectory_nodes, replace=False)
        members = sorted([members[int(i)] for i in positions], key=lambda n: (n.time, n.sample_id))
    times = [n.time for n in members]
    batches = []
    for node in members:
        positions = rng.choice(len(node.rows), size=t.cell_batch_size, replace=len(node.rows) < t.cell_batch_size)
        batches.append((node.latent[positions].to(pipeline.device), node.codes[positions].to(pipeline.device), node.rows[positions]))
    pairs = training_pairs(times, t.sources_per_target, rng)
    if not pairs:
        return None
    triples = endpoint_triples(times, t.endpoint_pairs, rng) if t.endpoint_weight else []
    # A bounded trajectory cache avoids one sparse slice per supervised pair.
    expressions = tensor_rows(adata, pipeline, np.concatenate([b[2] for b in batches]))
    expressions = expressions.split(t.cell_batch_size)
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for source, target in pairs:
        z, codes, _ = batches[source]
        predicted = flow(pipeline, times[source], times[target], z, codes)
        loss = predicted.sum() * 0.0
        if t.latent_ot_weight:
            loss = loss + t.latent_ot_weight * sinkhorn(predicted, batches[target][0])
        if t.gene_ot_weight:
            loss = loss + t.gene_ot_weight * sinkhorn(expressions[target], decode(pipeline, predicted))
        loss = loss / len(pairs)
        if not torch.isfinite(loss):
            raise RuntimeError("Transport pair loss became nonfinite")
        total += float(loss.detach())
        loss.backward()
    for a, b, target in triples:
        za, ca, _ = batches[a]
        zb, cb, _ = batches[b]
        first = flow(pipeline, times[a], times[target], za, ca)
        second = flow(pipeline, times[b], times[target], zb, cb)
        loss = t.endpoint_weight * sinkhorn(first, second) / len(triples)
        if not torch.isfinite(loss):
            raise RuntimeError("Endpoint consistency became nonfinite")
        total += float(loss.detach())
        loss.backward()
    torch.nn.utils.clip_grad_norm_(pipeline.transport.parameters(), 5.0)
    optimizer.step()
    return total


def validation_score(adata, pipeline, training_nodes, validation_nodes):
    """Score complete held-out donors from pooled youngest-age training cells."""
    from .selection import evaluate_holdout_l_gene, pooled_youngest_forward
    source_age = min(node.time for node in training_nodes)
    pool = [node for node in training_nodes if node.time == source_age]
    latent = torch.cat([node.latent for node in pool])
    codes = torch.cat([node.codes for node in pool])
    entries = {source_age: np.arange(len(latent))}
    donors = defaultdict(list)
    for node in validation_nodes:
        donors[node.sample_id].append(node)
    groups = [(donor, nodes[0].time, np.sort(np.concatenate([n.rows for n in nodes])))
              for donor, nodes in sorted(donors.items())]

    def rollout(age, rows, count, target_age):
        moved = flow(pipeline, age, target_age, latent[rows].to(pipeline.device), codes[rows].to(pipeline.device))
        return decode(pipeline, moved)

    d, t = pipeline.config.data, pipeline.config.training
    result = evaluate_holdout_l_gene(adata, age_key=d.time_key, donor_key=d.sample_key,
        seed=t.seed, device=pipeline.device, groups=groups,
        get_expression=lambda data, rows: tensor_rows(data, pipeline, rows),
        predict_group=lambda age, rows, rng: pooled_youngest_forward(entries, age, len(rows), rng, rollout),
        blur=t.sinkhorn_blur, scaling=t.sinkhorn_scaling)
    return result["selection_score"]


def train_transport(adata, pipeline, nodes, validation_nodes):
    t = pipeline.config.training
    rng = np.random.default_rng(t.seed)
    families = defaultdict(list)
    for node in nodes:
        if len(node.rows) >= t.min_population_cells:
            families[node.family].append(node)
    families = {key: members for key, members in families.items()
                if len(members) >= t.min_population_nodes and len({n.time for n in members}) >= 2}
    if not families:
        raise ValueError("No training stratum/cell type has sufficient population support")
    pipeline.trained_families = set(families)
    groups = list(families.values())
    optimizer = torch.optim.Adam(pipeline.transport.parameters(), lr=t.flow_lr, betas=tuple(t.adam_betas))
    sinkhorn = distance(pipeline)
    best_score, best_state, best_epoch = float("inf"), None, None
    for epoch in range(1, t.flow_epochs + 1):
        pipeline.transport.train()
        totals = []
        for _ in range(t.steps_per_epoch):
            members = groups[int(rng.integers(len(groups)))]
            total = trajectory_step(adata, pipeline, members, rng, optimizer, sinkhorn)
            if total is not None:
                totals.append(total)
        pipeline.transport.eval()
        if not totals:
            raise RuntimeError("No supported temporal pairs were sampled this epoch")
        score = validation_score(adata, pipeline, nodes, validation_nodes) if t.selection == "validation_gene_ot" else None
        if t.selection == "last" or score < best_score:
            best_score = score if score is not None else float("inf")
            best_state = deepcopy(pipeline.transport.state_dict())
            best_epoch = epoch
        logger.info("Transport epoch %d/%d: loss=%.6g validation=%s", epoch, t.flow_epochs, float(np.mean(totals)), score)
    pipeline.transport.load_state_dict(best_state)
    pipeline.transport.eval()
    pipeline.selection = {"criterion": t.selection, "epoch": best_epoch,
                          "validation_score": None if t.selection == "last" else best_score,
                          "source_policy": "pooled_youngest_forward"}
    return pipeline.selection
