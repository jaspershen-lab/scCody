"""Shared held-out donor Sinkhorn scoring and pooled youngest-age rollout."""
import numpy as np
import pandas as pd
import torch
from .losses import sinkhorn_divergence


def pooled_youngest_forward(entries, target_age, count, rng, rollout):
    """Sample the pooled youngest training cells and roll directly to target age."""
    source_age = min(entries)
    pool = entries[source_age]
    selected = rng.choice(pool, size=count, replace=len(pool) < count)
    return rollout(source_age, selected, count, float(target_age))


def evaluate_holdout_l_gene(normal_holdout, *, age_key, donor_key, seed, device,
                            get_expression, predict_group, blur=0.05, scaling=0.5,
                            target_ages=(), min_cells=100, max_groups=80, max_cells=256,
                            groups=None):
    """Mean gene-space Sinkhorn over equally weighted internal holdout donors."""
    if groups is None:
        ages = pd.to_numeric(normal_holdout.obs[age_key]).to_numpy(dtype=float)
        donors = normal_holdout.obs[donor_key].astype(str).to_numpy()
        groups = [(donor, float(np.median(ages[donors == donor])), np.flatnonzero(donors == donor))
                  for donor in sorted(pd.unique(donors))]
    allowed = np.asarray(target_ages, dtype=float)
    groups = [(donor, age, np.asarray(rows, dtype=np.int64)) for donor, age, rows in groups
              if len(rows) >= min_cells and (not len(allowed) or np.any(np.isclose(allowed, age)))]
    rng = np.random.default_rng(int(seed))
    if max_groups > 0 and len(groups) > max_groups:
        groups = [groups[int(i)] for i in rng.choice(len(groups), max_groups, replace=False)]
    device = torch.device(device)
    scores = []
    with torch.no_grad():
        for _, target_age, rows in groups:
            if max_cells > 0 and len(rows) > max_cells:
                rows = rng.choice(rows, max_cells, replace=False)
            predicted = predict_group(target_age, rows, rng)
            if predicted is None or len(predicted) < 2:
                continue
            true = get_expression(normal_holdout, rows[:len(predicted)])
            value = sinkhorn_divergence(torch.as_tensor(true, dtype=torch.float32, device=device),
                                        torch.as_tensor(predicted, dtype=torch.float32, device=device),
                                        blur=blur, scaling=scaling)
            scores.append(float(value.detach().cpu()))
    mean = float(np.mean(scores)) if scores else float("inf")
    return {"selection_score": mean, "normal_holdout_mean_L_gene": mean,
            "normal_holdout_selection_n_groups": len(scores)}
