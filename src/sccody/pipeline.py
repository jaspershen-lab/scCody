"""Fit, save, reload, generate, and localize arbitrary sample populations."""
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import anndata as ad
from .config import PipelineConfig
from .data import align_genes, prepare, validate_data
from .models.representation import VAE, GMVAE, VAE_NB, GMVAE_NB
from .models.transport import FlowMapTimeEmbed
from .models.conditional import CellTypeConditionedFlowMap
from .training import decode, distance, flow, populations, sample_node, tensor_rows, train_representation, train_transport

logger = logging.getLogger(__name__)
SCHEMA_VERSION = 1


def resolve_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; set training.device=cpu or auto")
    return device


def write_json(path, values):
    path.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")


@dataclass
class LocalizationResult:
    predictions: pd.DataFrame
    curves: pd.DataFrame

    def save(self, outdir):
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=False)
        self.predictions.to_csv(outdir / "predictions.csv", index=False)
        self.curves.to_csv(outdir / "distance_curves.csv", index=False)


class Pipeline:
    """A frozen representation and a global or cell-type-conditioned transport model.

    ``fit`` writes a new run directory. ``load`` restores a self-contained checkpoint,
    including training reference populations, the gene panel, and time coordinates.
    """

    def __init__(self, config=None):
        self.config = PipelineConfig.from_mapping(config.to_dict()) if config is not None else PipelineConfig()
        self.config.validate()
        self.device = resolve_device(self.config.training.device)
        self.representation = None
        self.transport = None
        self.reference = []
        self.selection = None
        self.split_table = None

    def _build_models(self):
        m = self.config.model
        types = {"vae": VAE, "gmvae": GMVAE, "vae_nb": VAE_NB, "gmvae_nb": GMVAE_NB}
        kwargs = dict(latent_dim=m.latent_dim, hidden_dims=m.hidden_dims, dropout=m.dropout)
        if m.representation.startswith("gmvae"):
            kwargs["n_clusters"] = m.clusters
        self.representation = types[m.representation](len(self.genes), **kwargs).to(self.device)
        self.transport = FlowMapTimeEmbed(m.latent_dim, m.flow_hidden_dims).to(self.device)
        if m.scope == "cell_type":
            self.transport = CellTypeConditionedFlowMap(self.transport, len(self.cell_types), m.latent_dim,
                                                       mode="shared_plus_heads").to(self.device)

    def _bind_genes(self, adata):
        d, m = self.config.data, self.config.model
        if d.input_space == "counts" and not m.representation.endswith("_nb"):
            # Counts use the original library panel even if inference provides extra genes.
            self.library_indices = align_genes(adata, self.input_genes)
            self.gene_indices = pd.Index(self.input_genes).get_indexer(self.genes)
        else:
            self.library_indices = None
            self.gene_indices = align_genes(adata, self.genes)

    def fit(self, adata, outdir):
        """Prepare sample-disjoint data, fit representation, and select transport epochs."""
        if self.representation is not None:
            raise ValueError("Create a new Pipeline to fit another dataset")
        prepared = prepare(adata, self.config)
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=False)
        self.genes = prepared.genes
        self.input_genes = list(map(str, adata.var_names))
        self.cell_types = prepared.cell_types
        self.time_origin, self.time_scale = prepared.time_origin, prepared.time_scale
        self.split_table = prepared.split_table
        self._bind_genes(adata)
        self.config.save(outdir / "config.yaml")
        self.split_table.to_csv(outdir / "samples.csv", index=False)
        (outdir / "genes.txt").write_text("\n".join(self.genes) + "\n")
        write_json(outdir / "run.json", {"schema_version": SCHEMA_VERSION, "status": "training",
                                        "train_cells": len(prepared.train_indices),
                                        "validation_cells": len(prepared.validation_indices),
                                        "test_cells": len(prepared.test_indices)})
        try:
            torch.manual_seed(self.config.training.seed)
            self._build_models()
            train_representation(adata, self, prepared.train_indices, outdir)
            train_nodes = populations(adata, self, prepared.train_indices)
            validation_nodes = populations(adata, self, prepared.validation_indices, unit="sample") if self.config.training.selection != "last" else []
            train_transport(adata, self, train_nodes, validation_nodes, outdir)
            rng = np.random.default_rng(self.config.training.seed + 2)
            for node in train_nodes:
                z, codes, rows = sample_node(node, self.config.training.max_reference_cells, rng, "cpu")
                self.reference.append({"time": node.time, "stratum": node.stratum,
                                       "cell_type": node.cell_type, "latent": z, "codes": codes,
                                       "cell_ids": list(map(str, adata.obs_names[rows]))})
            self.save(outdir / "model.pt")
            write_json(outdir / "run.json", {"schema_version": SCHEMA_VERSION, "status": "complete",
                                            "selection": self.selection,
                                            "time_origin": self.time_origin, "time_scale": self.time_scale,
                                            "train_cells": len(prepared.train_indices),
                                            "validation_cells": len(prepared.validation_indices),
                                            "test_cells": len(prepared.test_indices),
                                            "gene_panel_sha256": hashlib.sha256("\n".join(self.genes).encode()).hexdigest(),
                                            "checkpoint_sha256": hashlib.sha256((outdir / "model.pt").read_bytes()).hexdigest()})
        except Exception as exc:
            write_json(outdir / "run.json", {"schema_version": SCHEMA_VERSION, "status": "failed", "error": str(exc)})
            raise
        return self

    def _check_fitted(self):
        if self.representation is None or self.selection is None:
            raise ValueError("Fit or load a pipeline first")

    def save(self, path):
        """Save a new-format checkpoint; original research checkpoints need migration."""
        self._check_fitted()
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
        payload = {"schema_version": SCHEMA_VERSION, "config": self.config.to_dict(),
                   "genes": self.genes, "input_genes": self.input_genes, "cell_types": self.cell_types,
                   "time_origin": self.time_origin, "time_scale": self.time_scale,
                   "representation": {k: v.detach().cpu() for k, v in self.representation.state_dict().items()},
                   "transport": {k: v.detach().cpu() for k, v in self.transport.state_dict().items()},
                   "reference": self.reference, "selection": self.selection,
                   "samples": self.split_table.to_dict(orient="list")}
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, *, device="auto"):
        path = Path(path)
        if path.is_dir():
            path = path / "model.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported checkpoint format; use a checkpoint produced by this scCody pipeline")
        config = PipelineConfig.from_mapping(payload["config"])
        config.training.device = device
        pipeline = cls(config)
        for key in ("genes", "input_genes", "cell_types", "time_origin", "time_scale", "reference", "selection"):
            setattr(pipeline, key, payload[key])
        pipeline.split_table = pd.DataFrame(payload["samples"])
        pipeline._build_models()
        pipeline.representation.load_state_dict(payload["representation"])
        pipeline.transport.load_state_dict(payload["transport"])
        pipeline.representation.eval().requires_grad_(False)
        pipeline.transport.eval()
        return pipeline

    @torch.no_grad()
    def generate(self, source, target_time):
        """Transport each source cell to a known time and decode selected genes."""
        self._check_fitted()
        validate_data(source, self.config, require_sample=False)
        self._bind_genes(source)
        if not np.isfinite(target_time):
            raise ValueError("target_time must be finite")
        d, t = self.config.data, self.config.training
        times = pd.to_numeric(source.obs[d.time_key]).to_numpy(dtype=float)
        if self.config.model.scope == "cell_type":
            codes = pd.Index(self.cell_types).get_indexer(source.obs[d.cell_type_key].astype(str))
            if (codes < 0).any():
                raise ValueError("Source contains a cell type absent from training")
        else:
            codes = np.zeros(source.n_obs, dtype=int)
        output = np.empty((source.n_obs, len(self.genes)), dtype=np.float32)
        # Retain original cell order while handling mixed source times.
        for source_time in np.unique(times):
            rows = np.flatnonzero(times == source_time)
            for start in range(0, len(rows), t.representation_batch_size):
                batch = rows[start:start + t.representation_batch_size]
                z = self.representation.encode(tensor_rows(source, self, batch))[0]
                c = torch.as_tensor(codes[batch], device=self.device, dtype=torch.long)
                output[batch] = decode(self, flow(self, source_time, target_time, z, c)).cpu().numpy()
        obs = source.obs.copy()
        # Keep h5ad output portable with pandas' optional Arrow string backend.
        for key in obs.columns:
            if isinstance(obs[key].dtype, pd.CategoricalDtype):
                categories = obs[key].cat.categories
                if pd.api.types.is_string_dtype(categories.dtype):
                    obs[key] = obs[key].cat.rename_categories(pd.Index(categories, dtype=object))
            elif pd.api.types.is_string_dtype(obs[key].dtype):
                obs[key] = pd.Categorical(obs[key], categories=pd.Index(obs[key].dropna().unique(), dtype=object))
        obs["_sccody_source_time"] = times
        obs[d.time_key] = float(target_time)
        obs.index = pd.Index(obs.index, dtype=object)
        result = ad.AnnData(output, obs=obs, var=pd.DataFrame(index=pd.Index(self.genes, dtype=object)))
        result.uns["sccody"] = {"time_origin": self.time_origin, "time_scale": self.time_scale,
                                "selected_epoch": self.selection["epoch"],
                                "expression_space": "count_mean" if self.config.model.representation.endswith("_nb") else "log1p"}
        return result

    @torch.no_grad()
    def localize(self, target, candidate_times, *, source_times=None, max_cells=512, min_cells=5):
        """Infer time from expression; target time labels are never read.

        Average source estimates within a sample/stratum/cell-type population. Each
        source-time pool is drawn from training cells stored in the checkpoint.
        """
        self._check_fitted()
        validate_data(target, self.config, require_time=False)
        self._bind_genes(target)
        grid = np.asarray(candidate_times, dtype=float)
        if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or len(np.unique(grid)) != len(grid):
            raise ValueError("candidate_times must be a nonempty sequence of distinct finite numbers")
        grid = np.sort(grid)
        if not isinstance(max_cells, int) or not isinstance(min_cells, int) or min_cells < 1 or max_cells < min_cells:
            raise ValueError("Require integer max_cells >= min_cells >= 1")
        requested = None if source_times is None else np.asarray(source_times, dtype=float)
        if requested is not None and (requested.ndim != 1 or not len(requested) or not np.isfinite(requested).all() or len(np.unique(requested)) != len(requested)):
            raise ValueError("source_times must contain distinct finite numbers")
        d = self.config.data
        keys = list(dict.fromkeys([d.sample_key, *d.stratify_keys,
                                 *([d.cell_type_key] if self.config.model.scope == "cell_type" else [])]))
        metadata = target.obs[keys].copy()
        for key in keys:
            metadata[key] = metadata[key].astype(str)
        metadata["_row"] = np.arange(target.n_obs)
        rng = np.random.default_rng(self.config.training.seed)
        sinkhorn = distance(self)
        predictions, curves = [], []
        for _, group in metadata.groupby(keys, observed=True, sort=True):
            info = {key: str(group[key].iloc[0]) for key in keys}
            if len(group) < min_cells:
                predictions.append({**info, "predicted_time": np.nan, "n_cells": len(group), "n_sources": 0, "status": "too_few_cells"})
                continue
            stratum = tuple(info[k] for k in d.stratify_keys)
            ct = info[d.cell_type_key] if self.config.model.scope == "cell_type" else None
            matching = [r for r in self.reference if r["stratum"] == stratum and r["cell_type"] == ct]
            available = sorted({r["time"] for r in matching})
            ages = available if requested is None else list(requested)
            if not ages or any(time not in available for time in ages):
                predictions.append({**info, "predicted_time": np.nan, "n_cells": len(group), "n_sources": 0, "status": "unsupported_source"})
                continue
            # Select sources and cells independently of target time labels and candidate scores.
            rows = rng.choice(group["_row"].to_numpy(), min(max_cells, len(group)), replace=False)
            target_z = self.representation.encode(tensor_rows(target, self, rows))[0]
            estimates = []
            for source_time in ages:
                members = [r for r in matching if r["time"] == source_time]
                source_z = torch.cat([r["latent"] for r in members])
                source_codes = torch.cat([r["codes"] for r in members])
                subset = rng.choice(len(source_z), min(max_cells, len(source_z)), replace=False)
                source_z = source_z[subset].to(self.device)
                source_codes = source_codes[subset].to(self.device)
                scores = np.array([float(sinkhorn(flow(self, source_time, time, source_z, source_codes), target_z).detach()) for time in grid])
                if not np.isfinite(scores).all():
                    raise RuntimeError("Localization produced nonfinite distances")
                # Fixed-seed ties do not depend on supplied chronological labels.
                ties = np.flatnonzero(np.isclose(scores, scores.min(), rtol=1e-7, atol=1e-9))
                chosen = float(grid[int(rng.choice(ties))])
                estimates.append(chosen)
                curves.extend({**info, "source_time": source_time, "candidate_time": float(time),
                               "distance": float(score), "source_prediction": chosen} for time, score in zip(grid, scores))
            predictions.append({**info, "predicted_time": float(np.mean(estimates)), "n_cells": len(group),
                                "n_sources": len(estimates), "status": "ok"})
        pred_columns = [*keys, "predicted_time", "n_cells", "n_sources", "status"]
        curve_columns = [*keys, "source_time", "candidate_time", "distance", "source_prediction"]
        return LocalizationResult(pd.DataFrame(predictions, columns=pred_columns), pd.DataFrame(curves, columns=curve_columns))
