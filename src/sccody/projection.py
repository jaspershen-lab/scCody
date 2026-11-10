"""Aggregation of generative trajectory localization across independent fits."""
import numpy as np
import pandas as pd


def localize_ensemble(models, target, candidate_times, *, source_times=None, **kwargs):
    """Average source estimates within each fit, then equally across all fits.

    Every fit must support a population. Missing fits produce an unsupported
    result rather than silently changing the ensemble. Models retain their
    own latent spaces, reference populations and source-age selections.
    ``source_times`` is a list per fit, or None to use each model's references.
    """
    models = list(models)
    if not models:
        raise ValueError("Provide at least one fitted model")
    candidate_times = list(candidate_times)
    if source_times is not None and len(source_times) != len(models):
        raise ValueError("source_times must contain one list per fit")
    first = models[0].config.data
    scope = kwargs.get("population_scope") or models[0].config.model.scope
    keys = list(dict.fromkeys([first.sample_key, *first.stratify_keys,
                              *([first.cell_type_key] if scope == "cell_type" else [])]))
    predictions, curves = [], []
    for fit, model in enumerate(models):
        d = model.config.data
        if (d.sample_key, d.cell_type_key, d.stratify_keys) != (first.sample_key, first.cell_type_key, first.stratify_keys):
            raise ValueError("Ensemble metadata columns must agree")
        result = model.localize(target, candidate_times,
                                source_times=None if source_times is None else source_times[fit],
                                **{**kwargs, "population_scope": scope})
        predictions.append(result.predictions.assign(fit=fit))
        curves.append(result.curves.assign(fit=fit))
    per_fit = pd.concat(predictions, ignore_index=True)
    rows = []
    for _, group in per_fit.groupby(keys, observed=True, sort=True):
        complete = len(group) == len(models) and (group.status == "ok").all()
        rows.append({**{key: group[key].iloc[0] for key in keys},
                     "predicted_time": float(group.predicted_time.mean()) if complete else np.nan,
                     "n_fits": int((group.status == "ok").sum()),
                     "status": "ok" if complete else "unsupported_fit"})
    from .model import LocalizationResult
    return LocalizationResult(pd.DataFrame(rows, columns=[*keys, "predicted_time", "n_fits", "status"]),
                              pd.concat(curves, ignore_index=True))
