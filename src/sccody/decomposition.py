"""Disease fields, healthy-aging projections and donor-weighted heterogeneity.

"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def exact_assignment_columns(source, target):
    """Minimum-cost permutation for equal uniform latent populations."""
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if source.ndim != 2 or source.shape != target.shape or not len(source):
        raise ValueError("Exact assignment requires equal nonempty (cells, dimensions) arrays")
    cost = np.maximum(np.sum(source * source, axis=1, keepdims=True)
                      + np.sum(target * target, axis=1, keepdims=True).T
                      - 2.0 * source @ target.T, 0.0)
    rows, columns = linear_sum_assignment(cost)
    return columns[np.argsort(rows)]

def exact_assignment_disease_field(
    source: np.ndarray,
    target: np.ndarray,
    source_cell_types: np.ndarray | None = None,
    target_cell_types: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute the exact empirical W2 map for equal uniform populations.

    Equal source and target cardinality makes the balanced discrete OT optimum
    a minimum-cost permutation.  The result is a deterministic map rather than
    a diffuse barycentric approximation.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    target_columns = exact_assignment_columns(source, target)
    mapped = target[target_columns]
    displacement = mapped - source
    same_type_fraction = np.nan
    if source_cell_types is not None and target_cell_types is not None:
        source_types = np.asarray(source_cell_types)
        target_types = np.asarray(target_cell_types)
        if len(source_types) != len(source) or len(target_types) != len(target):
            raise ValueError("Cell-type labels must align with source and target cells")
        same_type_fraction = float(np.mean(source_types == target_types[target_columns]))
    disease_energy = float(np.mean(np.sum(displacement * displacement, axis=1)))
    diagnostics = {
        "conditional_transport_variance": 0.0,
        "barycentric_ot_cost": disease_energy,
        "normalized_row_entropy": 0.0,
        "target_marginal_l1_error": 0.0,
        "target_marginal_max_error": 0.0,
        "same_cell_type_transport_mass_fraction": same_type_fraction,
    }
    return displacement.astype(np.float32), diagnostics

def alignment_from_moments(
    *,
    inner_product: float,
    aging_energy: float,
    disease_energy: float,
    fixed_residual_energy: float,
    conditional_transport_variance: float = np.nan,
    energy_tolerance: float = 1e-12,
) -> dict[str, float | str]:
    """Compute parallel/perpendicular metrics from population moments."""
    inner = float(inner_product)
    aging = float(aging_energy)
    disease = float(disease_energy)
    fixed_residual = float(fixed_residual_energy)
    conditional_variance = float(conditional_transport_variance)
    result: dict[str, float | str] = {
        "inner_product": inner,
        "aging_energy": aging,
        "disease_energy": disease,
        "fixed_residual_energy": fixed_residual,
        "conditional_transport_variance": conditional_variance,
        "aging_rms_displacement": float(np.sqrt(max(aging, 0.0))),
        "disease_rms_displacement": float(np.sqrt(max(disease, 0.0))),
    }
    if not np.isfinite(disease) or disease <= energy_tolerance:
        result.update(
            {
                "alignment_status": "disease_field_near_zero",
                "alpha": np.nan,
                "cosine_alignment": np.nan,
                "parallel_energy": np.nan,
                "perpendicular_energy": np.nan,
                "aging_energy_fraction": np.nan,
                "signed_aging_energy_fraction": np.nan,
                "perpendicular_energy_fraction": np.nan,
                "fixed_map_explained_fraction": np.nan,
            }
        )
    elif not np.isfinite(aging) or aging <= energy_tolerance:
        result.update(
            {
                "alignment_status": "aging_field_near_zero",
                "alpha": np.nan,
                "cosine_alignment": np.nan,
                "parallel_energy": np.nan,
                "perpendicular_energy": disease,
                "aging_energy_fraction": np.nan,
                "signed_aging_energy_fraction": np.nan,
                "perpendicular_energy_fraction": 1.0,
                "fixed_map_explained_fraction": float(1.0 - fixed_residual / disease),
            }
        )
    else:
        alpha = inner / aging
        cosine = inner / np.sqrt(aging * disease)
        cosine = float(np.clip(cosine, -1.0, 1.0))
        parallel = max(0.0, inner * inner / aging)
        parallel = min(parallel, disease)
        perpendicular = max(0.0, disease - parallel)
        fraction = float(np.clip(parallel / disease, 0.0, 1.0))
        result.update(
            {
                "alignment_status": "ok",
                "alpha": float(alpha),
                "cosine_alignment": cosine,
                "parallel_energy": float(parallel),
                "perpendicular_energy": float(perpendicular),
                "aging_energy_fraction": fraction,
                "signed_aging_energy_fraction": float(np.sign(inner) * fraction),
                "perpendicular_energy_fraction": float(perpendicular / disease),
                "fixed_map_explained_fraction": float(1.0 - fixed_residual / disease),
            }
        )
    total_ot = disease + conditional_variance
    if np.isfinite(conditional_variance) and total_ot > energy_tolerance:
        result["barycentric_map_fidelity"] = float(disease / total_ot)
        result["conditional_variance_fraction"] = float(conditional_variance / total_ot)
    else:
        result["barycentric_map_fidelity"] = np.nan
        result["conditional_variance_fraction"] = np.nan
    return result

def displacement_alignment(
    disease_displacement: np.ndarray,
    aging_displacement: np.ndarray,
    *,
    conditional_transport_variance: float = np.nan,
) -> dict[str, float | str]:
    """Compute alignment metrics for aligned per-source-cell displacement fields."""
    disease = np.asarray(disease_displacement, dtype=np.float64)
    aging = np.asarray(aging_displacement, dtype=np.float64)
    if disease.ndim != 2 or aging.ndim != 2 or disease.shape != aging.shape:
        raise ValueError("Disease and aging fields must be aligned (cells, dimensions) arrays")
    if len(disease) == 0:
        raise ValueError("Displacement fields must contain at least one source cell")
    inner = float(np.mean(np.sum(disease * aging, axis=1)))
    aging_energy = float(np.mean(np.sum(aging * aging, axis=1)))
    disease_energy = float(np.mean(np.sum(disease * disease, axis=1)))
    fixed_residual = float(np.mean(np.sum((disease - aging) ** 2, axis=1)))
    return alignment_from_moments(
        inner_product=inner,
        aging_energy=aging_energy,
        disease_energy=disease_energy,
        fixed_residual_energy=fixed_residual,
        conditional_transport_variance=conditional_transport_variance,
    )


def match_populations(source, target, *, method="exact", blur=0.05, scaling=0.5):
    """Map a healthy latent population to a disease population.

    Exact matching requires equal cardinalities and returns the minimum-cost
    permutation. Barycentric matching permits unequal cardinalities and uses
    the research Sinkhorn conditional coupling. Match within the intended
    coarse cell group; labels do not otherwise constrain the assignment.
    """
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if (source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]
            or min(len(source), len(target)) == 0
            or not np.isfinite(source).all() or not np.isfinite(target).all()):
        raise ValueError("Provide finite nonempty populations with aligned latent dimensions")
    if method == "exact":
        field, diagnostics = exact_assignment_disease_field(source, target)
        return source + field, diagnostics
    if method != "barycentric":
        raise ValueError("method must be exact or barycentric")
    import torch
    from .losses import sinkhorn_plan
    with torch.no_grad():
        x, y = torch.as_tensor(source, dtype=torch.float32), torch.as_tensor(target, dtype=torch.float32)
        plan = sinkhorn_plan(x, y, blur=blur, scaling=scaling).numpy().astype(np.float64)
    conditional = plan / plan.sum(axis=1, keepdims=True)
    mapped = conditional @ target
    variance = np.maximum(conditional @ np.sum(target**2, axis=1) - np.sum(mapped**2, axis=1), 0).mean()
    cost = np.sum(plan * np.sum((source[:, None] - target[None])**2, axis=2))
    return mapped, {"conditional_transport_variance": float(variance),
                    "barycentric_ot_cost": float(cost),
                    "target_marginal_l1_error": float(np.abs(plan.sum(0) - 1 / len(target)).sum())}


def orthogonal_energy(donor_mean_vectors, within_donor_energy):
    """Equal-patient shared + between + within orthogonal energy identity."""
    vectors = np.asarray(donor_mean_vectors, dtype=np.float64)
    within = np.asarray(within_donor_energy, dtype=np.float64)
    if vectors.ndim != 2 or not len(vectors) or within.shape != (len(vectors),):
        raise ValueError("Provide one mean vector and within energy per patient")
    shared = vectors.mean(0)
    shared_energy = float(shared @ shared)
    between = float(np.sum((vectors - shared)**2, axis=1).mean())
    within_energy = float(within.mean())
    total = shared_energy + between + within_energy
    return {"mean_shift_energy": shared_energy, "between_donor_energy": between,
            "within_donor_energy": within_energy, "orthogonal_energy": total,
            "mean_shift_fraction": shared_energy / total if total > 0 else np.nan,
            "between_donor_fraction": between / total if total > 0 else np.nan,
            "within_donor_fraction": within_energy / total if total > 0 else np.nan,
            "orthogonal_rms": float(np.sqrt(max(total, 0)))}


def decompose_displacements(disease, aging, *, pair_ids=None, donor_ids=None, pair_alphas=None):
    """Project each matched pair field onto its healthy-aging field.

    Arrays have shape (matched cells, latent dimensions). IDs have one value
    per cell; each pair belongs to one target patient. Cells are averaged
    within pairs, pairs within patients, and patients equally. Within-patient
    energy includes variation between reference pairs. ``pair_alphas`` may
    supply recorded coefficients keyed by string pair ID for saved-field replay.
    A zero aging field
    uses alpha=0, as in the residual analysis, and is flagged in pair metrics.
    """
    disease, aging = np.asarray(disease, dtype=np.float64), np.asarray(aging, dtype=np.float64)
    if disease.ndim != 2 or disease.shape != aging.shape or not len(disease):
        raise ValueError("Provide aligned nonempty disease and aging fields")
    if not np.isfinite(disease).all() or not np.isfinite(aging).all():
        raise ValueError("Displacements must be finite")
    pair_ids = np.full(len(disease), "pair") if pair_ids is None else np.asarray(pair_ids, dtype=str)
    donor_ids = np.full(len(disease), "patient") if donor_ids is None else np.asarray(donor_ids, dtype=str)
    if pair_ids.shape != (len(disease),) or donor_ids.shape != pair_ids.shape:
        raise ValueError("IDs must have one value per matched source cell")
    aligned = np.empty_like(disease)
    pair_rows = []
    for pair in np.unique(pair_ids):
        mask = pair_ids == pair
        patients = np.unique(donor_ids[mask])
        if len(patients) != 1:
            raise ValueError("Each pair must belong to one patient")
        metrics = displacement_alignment(disease[mask], aging[mask])
        inner, energy = metrics["inner_product"], metrics["aging_energy"]
        alpha = inner / energy if energy > 1e-12 else 0.0
        if pair_alphas is not None:
            alpha = float(pair_alphas[pair])
            if not np.isfinite(alpha):
                raise ValueError("Recorded pair coefficients must be finite")
        aligned[mask] = alpha * aging[mask]
        metrics.update(pair_id=pair, target_donor=patients[0], alpha=alpha,
                       signed_aligned_rms=inner / np.sqrt(energy) if energy > 1e-12 else np.nan)
        pair_rows.append(metrics)
    residual = disease - aligned
    donors = np.unique(donor_ids)
    means, energies = [], []
    for donor in donors:
        pairs = np.unique(pair_ids[donor_ids == donor])
        means.append(np.mean([residual[pair_ids == pair].mean(0) for pair in pairs], axis=0))
        energies.append(np.mean([np.sum(residual[pair_ids == pair]**2, axis=1).mean() for pair in pairs]))
    means = np.stack(means)
    shared = means.mean(0)
    donor_mean = means[np.searchsorted(donors, donor_ids)]
    within = residual - donor_mean
    within_energy = np.maximum(np.asarray(energies) - np.sum(means**2, axis=1), 0)
    summary = orthogonal_energy(means, within_energy)
    pair_table = pd.DataFrame(pair_rows)
    summary["signed_aligned_rms"] = float(pair_table.groupby("target_donor", observed=True).signed_aligned_rms.mean().mean())
    return {"aligned": aligned, "residual": residual, "shared": shared,
            "between": donor_mean - shared, "within": within,
            "donor_means": means, "donors": donors, "donor_ids": donor_ids,
            "pair_ids": pair_ids, "pairs": pair_table, "summary": summary,
            "donor_within_energy": within_energy}


def decompose_disease(model, healthy_latent, disease_latent, *, source_age, target_age,
                      cell_type_codes=None, method="exact"):
    """Match one patient/reference pair and compute aging on the same controls.

    Ages define the healthy trajectory interval explicitly. Cohort matching,
    age/sex calipers and patient selection remain in reproducibility workflows.
    """
    mapped, diagnostics = match_populations(healthy_latent, disease_latent, method=method,
                                            blur=model.config.training.sinkhorn_blur,
                                            scaling=model.config.training.sinkhorn_scaling)
    aged = model.transport_latent(healthy_latent, source_age, target_age,
                                  cell_type_codes=cell_type_codes)
    result = decompose_displacements(mapped - healthy_latent, aged - healthy_latent)
    result.update(source=np.asarray(healthy_latent), mapped=mapped, aging=aged - healthy_latent,
                  matching=diagnostics)
    return result


def decode_components(model, source_latent, components):
    """Decode the research aligned→shared→between→within telescoping path.

    Returns per-cell expression effects in checkpoint gene order, including
    the reverse shared/aligned order to quantify decoder path dependence.
    """
    z = np.asarray(source_latent, dtype=np.float64)
    a, shared = components["aligned"], components["shared"]
    donor = shared + components["between"]
    x0 = model.decode_latent(z)
    x1 = model.decode_latent(z + a)
    x2 = model.decode_latent(z + a + shared)
    x3 = model.decode_latent(z + a + donor)
    x4 = model.decode_latent(z + a + donor + components["within"])
    xs = model.decode_latent(z + shared)
    return {"base": x0, "aligned": x1 - x0, "shared": x2 - x1,
            "between": x3 - x2, "within": x4 - x3, "total": x4 - x0,
            "shared_alt": xs - x0, "aligned_alt": x2 - xs}
