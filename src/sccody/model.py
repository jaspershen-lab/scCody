"""Fit, save, reload, generate, and localize arbitrary sample populations."""
import json
from dataclasses import dataclass
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import anndata as ad
from .config import ScCodyConfig
from .data import align_genes, prepare, validate_data
from .gmvae import VAE, GMVAE
from .flowmap import FlowMapTimeEmbed, CellTypeConditionedFlowMap
from .training import decode, distance, flow, populations, sample_node, tensor_rows, train_representation, train_transport

logger = logging.getLogger(__name__)
SCHEMA_VERSION = 1


def resolve_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)




@dataclass
class LocalizationResult:
    predictions: pd.DataFrame
    curves: pd.DataFrame

    def save(self, outdir):
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=False)
        self.predictions.to_csv(outdir / "predictions.csv", index=False)
        self.curves.to_csv(outdir / "distance_curves.csv", index=False)


class ScCody:
    """A frozen representation and a global or cell-type-conditioned transport model.

    ``fit`` writes a new run directory. ``load`` restores a self-contained checkpoint,
    including training reference populations, the gene panel, and time coordinates.
    """

    def __init__(self, config=None):
        self.config = ScCodyConfig.from_mapping(config.to_dict()) if config is not None else ScCodyConfig()
        self.device = resolve_device(self.config.training.device)
        self.representation = None
        self.transport = None
        self.reference = []
        self.selection = None
        self.split_table = None
        self.preprocessing = {"input_space": self.config.data.input_space,
                              "normalize_total": self.config.data.normalize_total,
                              "normalization_panel": "input_genes"}

    def _build_models(self):
        m = self.config.model
        types = {"vae": VAE, "gmvae": GMVAE}
        kwargs = dict(latent_dim=m.latent_dim, hidden_dims=m.hidden_dims, dropout=m.dropout)
        if m.representation == "gmvae":
            kwargs["n_clusters"] = m.clusters
        self.representation = types[m.representation](len(self.genes), **kwargs).to(self.device)
        self.transport = FlowMapTimeEmbed(m.latent_dim, m.flow_hidden_dims).to(self.device)
        if m.scope == "cell_type":
            self.transport = CellTypeConditionedFlowMap(self.transport, len(self.cell_types), m.latent_dim).to(self.device)

    def _bind_genes(self, adata):
        d = self.config.data
        if d.input_space == "counts":
            # Counts use the original library panel even if inference provides extra genes.
            self.library_indices = align_genes(adata, self.input_genes)
            self.gene_indices = pd.Index(self.input_genes).get_indexer(self.genes)
        else:
            self.library_indices = None
            self.gene_indices = align_genes(adata, self.genes)

    def fit(self, adata, outdir="sccody_fit"):
        """Prepare sample-disjoint data, fit representation, and select transport epochs."""
        if self.representation is not None:
            raise ValueError("Create a new ScCody to fit another dataset")
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
        torch.manual_seed(self.config.training.seed)
        self._build_models()
        if self.config.training.representation_checkpoint:
            checkpoint = torch.load(self.config.training.representation_checkpoint, map_location="cpu", weights_only=True)
            if checkpoint.get("kind") != "representation" or checkpoint.get("genes") != self.genes:
                raise ValueError("Representation checkpoint kind or gene order does not match input")
            if "config" in checkpoint:
                saved = ScCodyConfig.from_mapping(checkpoint["config"])
                for key in ("representation", "latent_dim", "hidden_dims", "clusters", "dropout", "no_softplus_decode"):
                    if getattr(saved.model, key) != getattr(self.config.model, key):
                        raise ValueError(f"Representation checkpoint model setting differs: {key}")
                for key in ("input_space", "normalize_total", "layer"):
                    if getattr(saved.data, key) != getattr(self.config.data, key):
                        raise ValueError(f"Representation preprocessing differs: {key}")
                if checkpoint["input_genes"] != self.input_genes:
                    raise ValueError("Representation input library gene order differs")
            self.representation.load_state_dict(checkpoint["state_dict"])
            self.representation.eval().requires_grad_(False)
        else:
            train_representation(adata, self, prepared.train_indices)
        self.save_representation(outdir / "representation.pt")
        train_nodes = populations(adata, self, prepared.train_indices)
        validation_nodes = populations(adata, self, prepared.validation_indices, unit="sample") if self.config.training.selection != "last" else []
        train_transport(adata, self, train_nodes, validation_nodes)
        rng = np.random.default_rng(self.config.training.seed + 2)
        for node in train_nodes:
            if node.family not in self.trained_families:
                continue
            size = self.config.training.max_reference_cells or len(node.rows)
            z, codes, rows = sample_node(node, size, rng, "cpu")
            self.reference.append({"time": node.time, "stratum": node.stratum,
                                   "cell_type": node.cell_type, "latent": z, "codes": codes,
                                   "types": list(adata.obs.iloc[rows][self.config.data.cell_type_key].astype(str)) if self.config.data.cell_type_key else None,
                                   "cell_ids": list(map(str, adata.obs_names[rows]))})
        self.save(outdir / "model.pt")
        (outdir / "selection.json").write_text(json.dumps(self.selection, indent=2) + "\n")
        return self

    def _check_fitted(self):
        if self.representation is None or self.selection is None:
            raise ValueError("Fit or load a ScCody model first")

    def save_representation(self, path, *, overwrite=False):
        """Save frozen weights and preprocessing metadata for reuse in a new fit."""
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
        torch.save({"kind": "representation", "genes": self.genes,
                    "input_genes": self.input_genes, "config": self.config.to_dict(),
                    "state_dict": {k: v.detach().cpu() for k, v in self.representation.state_dict().items()},
                    }, path)

    @torch.no_grad()
    def encode(self, adata):
        """Deterministic posterior-mean embedding in the checkpoint gene order."""
        self._check_fitted()
        validate_data(adata, self.config, require_time=False, require_sample=False,
                      require_strata=False, require_cell_type=False)
        self._bind_genes(adata)
        size = self.config.training.representation_batch_size
        return np.concatenate([self.representation.encode(tensor_rows(adata, self, np.arange(start, min(start + size, adata.n_obs))))[0].cpu().numpy()
                               for start in range(0, adata.n_obs, size)])

    @torch.no_grad()
    def transport_latent(self, latent, source_time, target_time, *, cell_type_codes=None):
        """Transport latent cells using chronological ages and saved age scaling.

        CT codes are integer indices into ``cell_types``, one per source cell.
        Source and target ages are scalar; use separate calls for mixed ages.
        """
        self._check_fitted()
        values = np.asarray(latent, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.config.model.latent_dim or not np.isfinite(values).all():
            raise ValueError("Provide a finite (cells, latent_dim) array")
        if not np.isfinite(source_time) or not np.isfinite(target_time):
            raise ValueError("Source and target ages must be finite")
        codes = np.zeros(len(values), dtype=np.int64)
        if self.config.model.scope == "cell_type":
            raw = np.asarray(cell_type_codes)
            if raw.shape != (len(values),) or raw.dtype.kind not in "iu" or (raw < 0).any() or (raw >= len(self.cell_types)).any():
                raise ValueError("CT transport requires valid integer codes in checkpoint cell_types order")
            codes = raw.astype(np.int64)
        result = np.empty_like(values)
        size = self.config.training.representation_batch_size
        for start in range(0, len(values), size):
            z = torch.as_tensor(values[start:start + size], device=self.device)
            c = torch.as_tensor(codes[start:start + size], device=self.device)
            result[start:start + size] = flow(self, source_time, target_time, z, c).cpu().numpy()
        return result

    @torch.no_grad()
    def decode_latent(self, latent):
        """Decode latent cells in bounded batches, using saved expression policy."""
        self._check_fitted()
        values = np.asarray(latent, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.config.model.latent_dim or not np.isfinite(values).all():
            raise ValueError("Provide a finite (cells, latent_dim) array")
        output = np.empty((len(values), len(self.genes)), dtype=np.float32)
        size = self.config.training.representation_batch_size
        for start in range(0, len(values), size):
            output[start:start + size] = decode(self, torch.as_tensor(values[start:start + size], device=self.device)).cpu().numpy()
        return output

    def save(self, path, *, overwrite=False):
        """Save model weights, preprocessing, age coordinates and reference populations."""
        self._check_fitted()
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
        payload = {"format": "sccody", "schema_version": SCHEMA_VERSION, "config": self.config.to_dict(),
                   "genes": self.genes, "input_genes": self.input_genes, "cell_types": self.cell_types,
                   "time_origin": self.time_origin, "time_scale": self.time_scale,
                   "representation": {k: v.detach().cpu() for k, v in self.representation.state_dict().items()},
                   "transport": {k: v.detach().cpu() for k, v in self.transport.state_dict().items()},
                   "reference": self.reference, "selection": self.selection,
                   "preprocessing": self.preprocessing,
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
        if payload.get("format") != "sccody" or payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported checkpoint format; expected a canonical scCody checkpoint")
        config = ScCodyConfig.from_mapping(payload["config"])
        config.training.device = device
        pipeline = cls(config)
        for key in ("genes", "input_genes", "cell_types", "time_origin", "time_scale", "reference", "selection", "preprocessing"):
            setattr(pipeline, key, payload[key])
        pipeline.split_table = pd.DataFrame(payload["samples"])
        pipeline._build_models()
        pipeline.representation.load_state_dict(payload["representation"])
        pipeline.transport.load_state_dict(payload["transport"])
        pipeline.representation.eval().requires_grad_(False)
        pipeline.transport.eval()
        return pipeline

    def set_reference(self, adata):
        """Set healthy reference populations from the supplied donors.

        Any healthy donors with known ages may serve as the reference; the
        study used the checkpoint's recorded training donors. Supply the
        source-age pools chosen by the experiment, with matching preprocessing
        and metadata.
        """
        self._check_fitted()
        validate_data(adata, self.config)
        self._bind_genes(adata)
        self.reference = []
        for node in populations(adata, self, np.arange(adata.n_obs)):
            self.reference.append({"time": node.time, "stratum": node.stratum,
                                   "cell_type": node.cell_type, "latent": node.latent, "codes": node.codes,
                                   "types": list(adata.obs.iloc[node.rows][self.config.data.cell_type_key].astype(str)) if self.config.data.cell_type_key else None,
                                   "cell_ids": list(map(str, adata.obs_names[node.rows]))})
        return self

    @torch.no_grad()
    def generate(self, source, target_age):
        """Transport each source cell to a known time and decode selected genes."""
        self._check_fitted()
        validate_data(source, self.config, require_sample=False, require_strata=False)
        self._bind_genes(source)
        if not np.isfinite(target_age):
            raise ValueError("target_age must be finite")
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
                output[batch] = decode(self, flow(self, source_time, target_age, z, c)).cpu().numpy()
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
        obs[d.time_key] = float(target_age)
        obs.index = pd.Index(obs.index, dtype=object)
        result = ad.AnnData(output, obs=obs, var=pd.DataFrame(index=pd.Index(self.genes, dtype=object)))
        result.uns["sccody"] = {"time_origin": self.time_origin, "time_scale": self.time_scale,
                                "selected_epoch": self.selection["epoch"],
                                "expression_space": "log1p"}
        return result

    @torch.no_grad()
    def localize(self, target, candidate_times, *, source_times=None, max_cells=512, min_cells=5,
                 population_scope=None, balanced_cells=False, pooled_source_cap=False):
        """Infer time from expression; target time labels are never read.

        Average source estimates within a sample/stratum/cell-type population. Each
        source-time pool is drawn from healthy reference cells stored in the checkpoint.
        """
        self._check_fitted()
        if not self.reference:
            raise ValueError("This checkpoint has no healthy reference populations; call set_reference(healthy_adata) before localization")
        population_scope = population_scope or self.config.model.scope
        if population_scope not in {"global", "cell_type"}:
            raise ValueError("population_scope must be global or cell_type")
        if population_scope == "cell_type" and not self.config.data.cell_type_key:
            raise ValueError("cell_type readout requires data.cell_type_key")
        validate_data(target, self.config, require_time=False)
        self._bind_genes(target)
        grid = np.asarray(candidate_times, dtype=float)
        if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or len(np.unique(grid)) != len(grid):
            raise ValueError("candidate_times must be a nonempty sequence of distinct finite numbers")
        grid = np.sort(grid)
        if min_cells < 1 or max_cells < min_cells:
            raise ValueError("Require integer max_cells >= min_cells >= 1")
        requested = None if source_times is None else np.asarray(source_times, dtype=float)
        if requested is not None and (requested.ndim != 1 or not len(requested) or not np.isfinite(requested).all() or len(np.unique(requested)) != len(requested)):
            raise ValueError("source_times must contain distinct finite numbers")
        d = self.config.data
        keys = list(dict.fromkeys([d.sample_key, *d.stratify_keys,
                                 *([d.cell_type_key] if population_scope == "cell_type" else [])]))
        metadata = target.obs[keys].copy()
        for key in keys:
            metadata[key] = metadata[key].astype(str)
        metadata["_row"] = np.arange(target.n_obs)
        rng = np.random.default_rng(self.config.training.seed)
        sinkhorn = distance(self)
        predictions, curves = [], []
        source_cache = {}
        for _, group in metadata.groupby(keys, observed=True, sort=True):
            info = {key: str(group[key].iloc[0]) for key in keys}
            if len(group) < min_cells:
                predictions.append({**info, "predicted_time": np.nan, "n_cells": len(group), "n_sources": 0, "status": "too_few_cells"})
                continue
            stratum = tuple(info[k] for k in d.stratify_keys)
            ct = info[d.cell_type_key] if population_scope == "cell_type" else None
            matching = [r for r in self.reference if r["stratum"] == stratum and
                        (pooled_source_cap or ct is None or r["cell_type"] == ct or (r.get("types") is not None and ct in r["types"]))]
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
                cache_key = (stratum, None if pooled_source_cap else ct, float(source_time))
                if cache_key not in source_cache:
                    source_z = torch.cat([r["latent"] for r in members])
                    source_codes = torch.cat([r["codes"] for r in members])
                    source_types = np.concatenate([np.asarray(r["types"]) if r.get("types") is not None else np.full(len(r["latent"]), r["cell_type"], dtype=object) for r in members])
                    if ct is not None and not pooled_source_cap:
                        masks = [np.asarray(r["types"]) == ct if r.get("types") is not None else np.full(len(r["latent"]), r["cell_type"] == ct) for r in members]
                        mask = np.concatenate(masks)
                        source_z, source_codes = source_z[mask], source_codes[mask]
                        source_types = source_types[mask]
                    subset = rng.choice(len(source_z), min(max_cells, len(source_z)), replace=False)
                    source_cache[cache_key] = (source_z[subset], source_codes[subset], source_types[subset])
                cached_z, cached_codes, cached_types = source_cache[cache_key]
                if ct is not None and pooled_source_cap:
                    mask = cached_types == ct
                    cached_z, cached_codes = cached_z[mask], cached_codes[mask]
                if len(cached_z) < 2:
                    break
                source_z, source_codes = cached_z.to(self.device), cached_codes.to(self.device)
                comparison_z = target_z
                if balanced_cells:
                    # Equal support sizes reproduce donor Sinkhorn reporting.
                    rows = rng.choice(group["_row"].to_numpy(), min(max_cells, len(group)), replace=False)
                    comparison_z = self.representation.encode(tensor_rows(target, self, rows))[0]
                    size = min(len(source_z), len(comparison_z))
                    source_z, source_codes, comparison_z = source_z[:size], source_codes[:size], comparison_z[:size]
                scores = np.array([float(sinkhorn(flow(self, source_time, time, source_z, source_codes), comparison_z).detach()) for time in grid])
                if not np.isfinite(scores).all():
                    raise RuntimeError("Localization produced nonfinite distances")
                # Fixed-seed ties do not depend on supplied chronological labels.
                ties = np.flatnonzero(np.isclose(scores, scores.min(), rtol=1e-6, atol=1e-8))
                chosen = float(grid[int(rng.choice(ties))])
                estimates.append(chosen)
                curves.extend({**info, "source_time": source_time, "candidate_time": float(time),
                               "distance": float(score), "source_prediction": chosen} for time, score in zip(grid, scores))
            complete = len(estimates) == len(ages)
            predictions.append({**info, "predicted_time": float(np.mean(estimates)) if complete else np.nan, "n_cells": len(group),
                                "n_sources": len(estimates), "status": "ok" if complete else "unsupported_source"})
        pred_columns = [*keys, "predicted_time", "n_cells", "n_sources", "status"]
        curve_columns = [*keys, "source_time", "candidate_time", "distance", "source_prediction"]
        return LocalizationResult(pd.DataFrame(predictions, columns=pred_columns), pd.DataFrame(curves, columns=curve_columns))
