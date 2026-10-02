# scCody

Train a single-cell population transport model on your own AnnData, generate expression at a requested time, and infer an unknown population's time from expression. Time can represent age, experimental duration, or another numeric coordinate. Sample IDs identify biological replicates.

The pipeline provides VAE or GMVAE representations, optional negative-binomial count models, and global or cell-type-conditioned transport. It accepts your column names, gene panel, sample splits, and matching strata. A saved model contains everything needed for inference, including bounded training reference populations.

## Install and try it

Use Python 3.10 or newer:

```bash
python -m pip install -e .
sccody demo --outdir outputs/demo
```

The demo runs a small synthetic experiment on CPU and writes a checkpoint, generated AnnData, distance curves, and evaluation tables. Its metrics demonstrate the workflow; they are not a biological benchmark.

## Use your data

Input is a `.h5ad` file containing:

| Field | Requirement |
| --- | --- |
| `X` or configured layer | Raw nonnegative counts, or explicitly declared log1p expression |
| `obs[sample_key]` | Biological sample/replicate ID |
| `obs[time_key]` | One finite numeric time per sample for fitting and known-time generation |
| `obs[cell_type_key]` | Required for cell-type-conditioned models |
| `obs[stratify_keys]` | Optional matching columns, with one value per sample |
| `var_names`, `obs_names` | Unique gene and cell IDs |

Each training stratum/cell type must cover at least two times. Validation populations need matching training support. Localization accepts targets without time labels.

Create and edit a configuration:

```bash
sccody init-config pipeline.yaml
sccody validate --data cells.h5ad --config pipeline.yaml
```

Set `data.time_key` and `data.sample_key` to your columns. For processed expression, set `data.input_space: log1p`; counts are normalized to the configured library total and log-transformed for MSE representations. NB representations consume raw counts. `data.layer` selects an alternate expression matrix. `data.n_genes` selects training-variable genes; set it to `null` to keep every gene.

For reproducible biological holdouts, supply `split.validation_samples` and `split.test_samples` as strings. If either list is nonempty, both lists define the full holdout assignment and the fraction settings are ignored. Otherwise, seeded sample-level fractions are used. All remaining samples train the representation and flow. Keep samples from the same biological unit under the same sample ID.

Fit and evaluate using a candidate grid chosen before inspecting test labels:

```bash
sccody run --data cells.h5ad --config pipeline.yaml \
  --outdir outputs/my_run --candidate-times 0,12,24,36,48
```

For fitting alone, use `sccody train` with the same data, config, and outdir arguments. Run directories must be new. The default checkpoint criterion is known-time validation expression Sinkhorn distance; `training.selection: last` keeps the final epoch. Test samples never select checkpoints.

## Inference from a saved model

```bash
# The target file can omit time labels.
sccody localize --model outputs/my_run --data unknown.h5ad \
  --candidate-times 0,12,24,36,48 --outdir outputs/localization

# Source cells need their observed source times.
sccody generate --model outputs/my_run --data source.h5ad \
  --target-time 48 --output outputs/generated.h5ad

# Evaluation requires known time labels and should use heldout samples.
sccody evaluate --model outputs/my_run --data heldout.h5ad \
  --candidate-times 0,12,24,36,48 --outdir outputs/evaluation
```

All inference commands accept `--device cpu`, `--device cuda:0`, or `--device auto`. Localization uses training source-time pools matched on the configured strata and cell type. It reports an unsupported population explicitly. `--source-times 0,12` requests an exact source set; an unavailable source prevents a reduced-source estimate. Inspect `distance_curves.csv` alongside `predictions.csv`.

Generated `X` contains decoded log1p expression for MSE representations or expected counts for NB representations. It contains the selected genes in checkpoint order and preserves source cell IDs. Extra inference genes are ignored. Count normalization requires the original input library panel; processed-input and NB models require the selected checkpoint genes. Missing required genes produce an error rather than implicit imputation.

## Python API

```python
import anndata as ad
from sccody import Pipeline, PipelineConfig

config = PipelineConfig.load("pipeline.yaml")
model = Pipeline(config).fit(ad.read_h5ad("cells.h5ad"), "outputs/python_run")
model = Pipeline.load("outputs/python_run", device="cpu")

generated = model.generate(ad.read_h5ad("source.h5ad"), target_time=48)
generated.write_h5ad("outputs/generated.h5ad")

result = model.localize(ad.read_h5ad("unknown.h5ad"), [0, 12, 24, 36, 48])
result.save("outputs/localization")
```

## Outputs and development

Each fit writes `config.yaml`, `samples.csv`, `genes.txt`, representation/flow histories, `model.pt`, and `run.json` with selection metadata, time coordinates, and hashes. Evaluation writes localization curves/predictions, `metrics.json`, and sample-level generation-versus-observed mean-expression metrics in `reconstruction.csv`. Correlation is across genes for reconstruction and across sample populations for localization.

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

The compact suite uses two synthetic schemas and panels, dense/sparse input, all four representation models, and both flow scopes. See [methods and limits](docs/methods.md), [migration and test retirement](docs/migration.md), and the [example configuration](examples/pipeline.yaml). This repository targets reusable user workflows. Historical experiment reproduction stays in the source research repository.
