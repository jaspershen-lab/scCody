"""Descriptive sample-level evaluation, separate from inference and selection."""
import numpy as np
import pandas as pd
from .training import tensor_rows


def localization_metrics(predictions, truth, *, sample_key, time_key, cell_type_key=None):
    keys = [sample_key] + ([cell_type_key] if cell_type_key else [])
    truth = truth[keys + [time_key]].copy()
    for key in keys:
        truth[key] = truth[key].astype(str)
    if (truth.groupby(keys, observed=True)[time_key].nunique() > 1).any():
        raise ValueError("Truth must contain one time per sample/population")
    merged = predictions.merge(truth.drop_duplicates(keys), on=keys, validate="one_to_one")
    valid = merged[merged["status"] == "ok"].copy()
    if not len(valid):
        return {"n_populations": 0, "mae": None, "pearson": None}
    y = pd.to_numeric(valid[time_key]).to_numpy(dtype=float)
    p = valid["predicted_time"].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Truth times must be finite")
    correlation = float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 and np.ptp(y) > 0 and np.ptp(p) > 0 else None
    return {"n_populations": len(valid), "mae": float(np.mean(np.abs(p - y))), "pearson": correlation}


def reconstruction_metrics(pipeline, target):
    """Compare training-source generation to known-time target mean expression.

    One row per heldout sample (or sample/cell type). These are descriptive
    population reconstruction metrics; no fabricated donor replicates or p-values.
    """
    from .data import validate_data
    from .training import decode, flow
    import torch
    validate_data(target, pipeline.config)
    pipeline._bind_genes(target)
    d = pipeline.config.data
    keys = list(dict.fromkeys([d.sample_key, *d.stratify_keys,
                             *([d.cell_type_key] if pipeline.config.model.scope == "cell_type" else [])]))
    metadata = target.obs[keys + [d.time_key]].copy()
    metadata["_row"] = np.arange(target.n_obs)
    records = []
    with torch.no_grad():
        for _, group in metadata.groupby(keys, observed=True, sort=True):
            info = {k: str(group[k].iloc[0]) for k in keys}
            times = pd.to_numeric(group[d.time_key]).unique()
            if len(times) != 1:
                raise ValueError("Each target sample/population must have one time")
            time = float(times[0])
            stratum = tuple(info[k] for k in d.stratify_keys)
            ct = info[d.cell_type_key] if pipeline.config.model.scope == "cell_type" else None
            refs = [r for r in pipeline.reference if r["stratum"] == stratum and r["cell_type"] == ct]
            if not refs:
                records.append({**info, "status": "unsupported_source", "rmse": np.nan, "pearson": np.nan})
                continue
            # Fix a source using training support; known target time is the requested generation time.
            source_time = min(r["time"] for r in refs)
            refs = [r for r in refs if r["time"] == source_time]
            generated_means = []
            batch_size = pipeline.config.training.representation_batch_size
            for ref in refs:
                sums = np.zeros(len(pipeline.genes), dtype=np.float64)
                for start in range(0, len(ref["latent"]), batch_size):
                    z = ref["latent"][start:start + batch_size].to(pipeline.device)
                    codes = ref["codes"][start:start + batch_size].to(pipeline.device)
                    sums += decode(pipeline, flow(pipeline, source_time, time, z, codes)).sum(dim=0).cpu().numpy()
                generated_means.append(sums / len(ref["latent"]))
            pred_mean = np.mean(generated_means, axis=0)
            real_sum = np.zeros(len(pipeline.genes), dtype=np.float64)
            rows = group["_row"].to_numpy()
            for start in range(0, len(rows), batch_size):
                real_sum += tensor_rows(target, pipeline, rows[start:start + batch_size]).sum(dim=0).cpu().numpy()
            real_mean = real_sum / len(rows)
            r = float(np.corrcoef(pred_mean, real_mean)[0, 1]) if np.ptp(pred_mean) and np.ptp(real_mean) else np.nan
            records.append({**info, "status": "ok", "source_time": source_time, "target_time": time,
                            "n_cells": len(rows), "rmse": float(np.sqrt(np.mean((pred_mean - real_mean) ** 2))), "pearson": r})
    return pd.DataFrame(records)
