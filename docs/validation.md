# Validation on 2 October 2026

The local CPU suite passed all 14 collected cases across three test modules. The checks cover two synthetic gene panels and metadata schemas, all four representation models, both flow scopes, real optimizer updates, informative validation selection, checkpoint reload, gene alignment, input/export contracts, label-free localization, and complete source support.

```bash
mamba run -n omics env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

The package was installed into a repository-local virtual environment with existing scientific dependencies. Its `sccody` entry point was verified from outside the repository without `PYTHONPATH`. The installed demo completed and saved a selected checkpoint, generated AnnData, localization curves/predictions, reconstruction metrics, and run receipts under `outputs/demo/` (ignored by Git).

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/sccody demo --outdir outputs/demo
```

The demo's validation score changed across flow epochs and selected epoch 2 using validation data only. Synthetic metric values are execution examples, not performance claims. Runtime versions and the selection receipt are in [validation.json](validation.json).

The host had no accessible CUDA device. GPU execution and the configured GitHub Actions Python matrix were not run locally. No Slurm jobs were submitted, replaced, or cancelled. `scCodyFlow` source, working-tree changes, and historical results were left in place.
