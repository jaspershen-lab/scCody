"""Small synthetic example requiring no research data or downloads."""
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from .config import PipelineConfig


def synthetic_data(*, seed=0, panel=0):
    rng = np.random.default_rng(seed)
    # Distinct schemas, panels, time scales, and storage expose dataset assumptions.
    sample_key, time_key, ct_key = ("sample", "time", "cell_type") if panel == 0 else ("replicate_id", "stage_hours", "state")
    records, matrices = [], []
    n_genes = 12 if panel == 0 else 17
    for sample in range(12):
        time = float(sample % 3) * (1 if panel == 0 else 12)
        for ct in ("state_x", "state_y"):
            n = 8
            rate = np.linspace(1, 3, n_genes) + time / (1 if panel == 0 else 12) + (ct == "state_y")
            matrices.append(rng.poisson(rate, size=(n, n_genes)).astype(np.float32))
            records.extend({sample_key: f"replicate_{sample}", time_key: time, ct_key: ct,
                            "plate": f"plate_{sample // 6}"} for _ in range(n))
    obs = pd.DataFrame(records, index=pd.Index([f"cell_{i}" for i in range(len(records))], dtype=object))
    for key in (sample_key, ct_key, "plate"):
        obs[key] = pd.Categorical(obs[key], categories=pd.Index(sorted(obs[key].unique()), dtype=object))
    matrix = np.concatenate(matrices)
    data = ad.AnnData(matrix if panel == 0 else sparse.csr_matrix(matrix), obs=obs,
                     var=pd.DataFrame(index=pd.Index([f"feature_{panel}_{i}" for i in range(n_genes)], dtype=object)))
    config = PipelineConfig.from_mapping({
        "data": {"sample_key": sample_key, "time_key": time_key, "cell_type_key": ct_key,
                 "stratify_keys": ["plate"], "n_genes": 10},
        "split": {"validation_samples": ["replicate_9"], "test_samples": ["replicate_10", "replicate_11"]},
        "model": {"latent_dim": 3, "hidden_dims": [16], "clusters": 3, "flow_hidden_dims": [16], "dropout": 0},
        "training": {"representation_epochs": 2, "representation_batch_size": 32, "flow_epochs": 2,
                     "steps_per_epoch": 3, "cell_batch_size": 8, "max_reference_cells": 8, "device": "cpu"},
    })
    return data, config
