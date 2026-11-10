"""Training-only feature selection and sample-disjoint preparation."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import sparse


def validate_data(adata, config, *, require_time=True, require_sample=True, require_strata=True,
                  require_cell_type=True):
    config.validate()
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("Input must contain cells and genes")
    if not adata.var_names.is_unique or not adata.obs_names.is_unique:
        raise ValueError("Gene and cell IDs must be unique")
    d = config.data
    keys = list(d.stratify_keys) if require_strata else []
    if require_time:
        keys.append(d.time_key)
    if require_sample:
        keys.append(d.sample_key)
    if require_cell_type and config.model.scope == "cell_type":
        keys.append(d.cell_type_key)
    for key in set(keys):
        if adata.obs[key].isna().any():
            raise ValueError(f"Observation column {key!r} contains missing values")
    if require_time:
        times = pd.to_numeric(adata.obs[d.time_key], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(times).all():
            raise ValueError("Time values must be finite numbers")
    if require_sample and require_time:
        if (adata.obs.groupby(d.sample_key, observed=True)[d.time_key].nunique() > 1).any():
            raise ValueError(f"Each sample must have one {d.time_key!r} value")
    matrix = get_matrix(adata, d.layer)
    # Validate without densifying the matrix; all transforms operate on cell batches.
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expression must be finite and nonnegative")
    if d.input_space == "counts" and not np.allclose(values, np.rint(values), atol=1e-5, rtol=0):
        raise ValueError("input_space=counts requires integer-valued raw counts; use log1p for processed input")


def get_matrix(adata, layer):
    if layer is None:
        return adata.X
    return adata.layers[layer]


def transform(matrix, data_config):
    values = matrix.astype(np.float32)
    if data_config.input_space == "counts":
        totals = np.asarray(values.sum(axis=1)).ravel()
        factors = data_config.normalize_total / np.maximum(totals, 1)
        if sparse.issparse(values):
            values = values.multiply(factors[:, None]).tocsr()
            values.data = np.log1p(values.data)
        else:
            values = np.log1p(values * factors[:, None])
    return values


def expression_rows(adata, config, indices, gene_indices, library_indices=None):
    # Normalize against the full input library, then select checkpoint genes.
    values = get_matrix(adata, config.data.layer)[indices]
    if library_indices is not None:
        values = values[:, library_indices]
    values = transform(values, config.data)
    values = values[:, gene_indices]
    if sparse.issparse(values):
        values = values.toarray()
    return np.asarray(values, dtype=np.float32)


def align_genes(adata, genes):
    indices = adata.var_names.get_indexer(genes)
    if (indices < 0).any():
        missing = np.asarray(genes)[indices < 0]
        raise ValueError(f"Input is missing {len(missing)} checkpoint genes, including {missing[:5].tolist()}")
    return indices


@dataclass
class PreparedData:
    genes: list[str]
    gene_indices: np.ndarray
    split_table: pd.DataFrame
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    time_origin: float
    time_scale: float
    cell_types: list[str]


def prepare(adata, config):
    validate_data(adata, config)
    d, s, m, t = config.data, config.split, config.model, config.training
    metadata = adata.obs.copy()
    metadata[d.sample_key] = metadata[d.sample_key].astype(str)
    metadata[d.time_key] = pd.to_numeric(metadata[d.time_key]).astype(float)
    samples = np.array(sorted(metadata[d.sample_key].unique()))
    rng = np.random.default_rng(t.seed)
    if s.validation_samples or s.test_samples:
        validation, test = set(s.validation_samples), set(s.test_samples)
        unknown = (validation | test) - set(samples)
        if unknown:
            raise ValueError(f"Unknown heldout sample IDs: {sorted(unknown)}")
    else:
        shuffled = rng.permutation(samples)
        nv = max(1, round(len(samples) * s.validation_fraction)) if s.validation_fraction else 0
        nt = max(1, round(len(samples) * s.test_fraction)) if s.test_fraction else 0
        validation, test = set(shuffled[:nv]), set(shuffled[nv:nv + nt])
    train = set(samples) - validation - test
    if len(train) < 2:
        raise ValueError("At least two training samples are required; reduce holdout fractions or provide explicit splits")
    if t.selection == "validation_gene_ot" and not validation:
        raise ValueError("validation_gene_ot selection requires validation samples; use selection=last for no selection")
    table = metadata[[d.sample_key, d.time_key, *d.stratify_keys]].drop_duplicates(d.sample_key).copy()
    table["split"] = table[d.sample_key].map(lambda x: "validation" if x in validation else "test" if x in test else "train")
    indices = {name: np.flatnonzero(metadata[d.sample_key].isin(ids).to_numpy())
               for name, ids in [("train", train), ("validation", validation), ("test", test)]}
    if d.max_train_time is not None:
        indices["train"] = indices["train"][metadata.iloc[indices["train"]][d.time_key].to_numpy() <= d.max_train_time]
        outside = (table["split"] == "train") & (table[d.time_key] > d.max_train_time)
        table.loc[outside, "split"] = "outside_fit_domain"
    if not len(indices["train"]):
        raise ValueError("No training cells remain in the configured fit domain")
    fitted_times = pd.to_numeric(metadata.iloc[indices["train"]][d.time_key]).to_numpy(dtype=float)
    if np.ptp(fitted_times) <= 0:
        raise ValueError("Training samples must cover at least two distinct times")
    genes = list(map(str, adata.var_names))
    if d.n_genes is not None and d.n_genes < adata.n_vars:
        # Compute moments in bounded batches, using only eligible training cells.
        total = np.zeros(adata.n_vars, dtype=np.float64)
        squares = np.zeros_like(total)
        for start in range(0, len(indices["train"]), t.representation_batch_size):
            block = get_matrix(adata, d.layer)[indices["train"][start:start + t.representation_batch_size]]
            block = transform(block, d).astype(np.float64)
            total += np.asarray(block.sum(axis=0)).ravel()
            squares += np.asarray(block.multiply(block).sum(axis=0) if sparse.issparse(block) else np.square(block).sum(axis=0)).ravel()
        variance = squares / len(indices["train"]) - np.square(total / len(indices["train"]))
        selected = np.sort(np.argsort(-variance, kind="stable")[:d.n_genes])
        genes = [genes[i] for i in selected]
    gene_indices = align_genes(adata, genes)
    origin = float(fitted_times.min()) if m.time_origin is None else float(m.time_origin)
    scale = float(np.ptp(fitted_times)) if m.time_scale is None else float(m.time_scale)
    observed_types = sorted(metadata.iloc[indices["train"]][d.cell_type_key].astype(str).unique()) if m.scope == "cell_type" else []
    cell_types = list(m.cell_types) if m.scope == "cell_type" and m.cell_types else observed_types
    if not set(observed_types) <= set(cell_types):
        raise ValueError("Configured cell_types omit types present in training")
    return PreparedData(genes, gene_indices, table, indices["train"], indices["validation"], indices["test"], origin, scale, cell_types)
