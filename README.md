# scCody

scCody is a generative framework for learning age-conditioned single-cell cohort dynamics from cross-sectional populations. It models healthy-aging trajectories and separates cellular changes aligned with those trajectories from off-trajectory dysregulation.

**scCody-G** learns a population-wide age-conditioned map. **scCody-CT** conditions transport on cell type. Both combine a Gaussian-mixture variational autoencoder (GMVAE) with FlowMap transport, population optimal-transport objectives, and endpoint consistency.

## Features

- Generate single-cell populations at a specified age.
- Localize donors along a healthy-aging trajectory using population distances.
- Decompose disease displacement into aging-aligned and orthogonal components, with shared, between-patient, and within-patient effects.
- Decode latent displacement components into expression changes.

## Installation

From a local checkout, with Python 3.10 or later:

```bash
pip install ./scCody
```

## Quick start

Download `scCody_weights.zip` from [Figshare](https://figshare.com/articles/software/scCody_Model_Weight/34951348?file=71249662) and extract it in the directory containing the sibling `scCody` and `scCody-reproducibility` checkouts. All study checkpoints are stored in `scCody-reproducibility/checkpoints/`; see the [checkpoint instructions](../scCody-reproducibility/manuscript/models/README.md). Run this example from the scCody repository root:

```python
import anndata as ad
from sccody import ScCody

model = ScCody.load("../scCody-reproducibility/checkpoints/AIDA_scCody-G_fit0.pt")
adata = ad.read_h5ad("data/healthy_cells.h5ad")
predicted = model.generate(adata, target_age=60)
predicted.write_h5ad("generated_age60.h5ad")
```

For the AIDA study checkpoints, provide log-normalized expression in `adata.X`, Ensembl gene IDs in `var_names`, and `age`, `sex`, and `Annotation_Level1` in `obs`; donor analyses also require `donor_id`. Use the checkpoint's ordered panel and preprocessing metadata. They expect CP10K normalization followed by `log1p` and panel selection. Generated values are decoded expression, not sampled raw counts.

## Training

```python
from sccody import ScCody, ScCodyConfig

config = ScCodyConfig.from_mapping({
    "data": {
        "time_key": "age", "sample_key": "donor_id",
        "cell_type_key": "Annotation_Level1", "input_space": "log1p",
    },
    "model": {"scope": "cell_type"},  # "global" for scCody-G
})
model = ScCody(config).fit(adata, outdir="my_fit")
model.save("model.pt")
```

Training holds out complete donors and stores the gene panel, age coordinates, weights, and healthy reference populations in `my_fit/model.pt`. By default, `training.selection="validation_gene_ot"` selects the epoch with the lowest donor-averaged gene-expression Sinkhorn divergence, rolling pooled youngest-age training cells forward to each validation donor’s age. The scorer uses at least 100 cells per donor, up to 80 donors and up to 256 cells per donor, shared with the companion comparators. The selected weights are saved in `model.pt`, with the final criterion, epoch and score in `selection.json`. Set `training.selection="last"` to keep the final epoch, including when training without validation donors. Configuration supports explicit donor splits and raw-count input. A saved checkpoint loads independently of its training directory.

## Inference and analysis

Donor-age localization uses expression and healthy reference populations; target chronological ages are not used to choose the inferred age:

```python
# For a newly fitted model, healthy references are already saved.
# The supplied pretrained weights need healthy references attached once:
model.set_reference(healthy_adata)
ages = model.localize(target_adata, candidate_times=range(19, 78))
print(ages.predictions)
model.save("model_with_references.pt")
```

`set_reference` accepts any healthy donors with known ages, prepared with the checkpoint's preprocessing and metadata. In the study, the healthy references were the training donors recorded in each checkpoint. `localize_ensemble` combines estimates from independently fitted models.

For one matched healthy/patient population, use the same cell group and explicit healthy-aging interval:

```python
from sccody import decompose_disease, decode_components

healthy = model.encode(healthy_adata)
patient = model.encode(patient_adata)
components = decompose_disease(
    model, healthy, patient, source_age=50, target_age=60,
    method="barycentric",
)
expression_effects = decode_components(model, healthy, components)
```

For multiple patients/reference pairs, `decompose_displacements` accepts matched disease and aging fields with pair and donor IDs, retaining equal-pair and equal-patient weighting. Cell-type-conditioned transport additionally requires integer `cell_type_codes` in `model.cell_types` order. Lower-level `encode`, `transport_latent`, and `decode_latent` methods expose the same fitted model.

## Pretrained models

| Models | Checkpoints | Fits |
|---|---|---|
| AIDA study models | `../scCody-reproducibility/checkpoints/AIDA_scCody-{G,CT}_fit{0,1,2}.pt` | 3 per variant |
| PsychAD disease-analysis models | `../scCody-reproducibility/checkpoints/PsychAD_scCody-{G,CT}_fit0.pt` | 1 per variant |

Each checkpoint contains the representation and transport weights together with its ordered gene panel, preprocessing settings, age scaling, and cell-type categories. The two model families were trained on different gene panels and preprocessing, so inputs should be prepared according to each checkpoint's metadata. Training details and the healthy-reference populations required for donor localization are described in the [checkpoint instructions](../scCody-reproducibility/manuscript/models/README.md).

## Reproducibility

The companion [scCody-reproducibility](../scCody-reproducibility) repository contains study preprocessing, donor splits, model selection, comparator benchmarks, disease analyses, and figure workflows.

## Citation

A publication citation is not yet available.
