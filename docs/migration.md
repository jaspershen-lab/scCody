# Migration to a reusable scCody repository

This new local repository was created from the reusable method core in `scCodyFlow` after reading `docs/results_final_provenance/REPORT.md`, audited 1 October 2026. The source repository and its working-tree changes remain in place. The report itself stays with the historical results. [Source receipts](source_receipt.json) record the source revision and hashes of the inspected report and extracted model files, including their uncommitted content.

The reusable elements are VAE/GMVAE representations and their count variants, identity-preserving flow maps, shared-trunk/cell-type-head transport, frozen representation training, gene/latent optimal transport, endpoint consistency, candidate-grid localization, and expression generation. The new orchestration has explicit sample-disjoint data contracts and a single configuration/API/CLI.

The report motivates preserving gene order, count/expression roles, model-time coordinates, physical biological holdouts, and actual checkpoint-selection metadata. It also motivates keeping label-free localization separate from known-time generation. Those are implemented as general contracts, rather than cohort-specific rules.

No AIDA, OneK1K, disease, comparator, paper-figure, or dated protocol module is included in the package. Gene panels, cell-type names, sample IDs, time grids, split seeds, and strata come from the user's input and configuration. Dataset-specific launchers, source manifests, controls, diagnostics, and result folders remain in the research repository.

## Test retirement

At extraction, the research working tree contained 46 test modules plus its `conftest.py`, with 216 named test functions. None were copied into this new repository. Dataset/protocol tests are retired from the user package, and generic legacy tests are replaced by tests of the new public contracts. The file-by-file decisions are recorded in [test_retirement.csv](test_retirement.csv). They do not delete or invalidate historical research tests.

The new suite has three test modules and a small synthetic fixture file. Four representation/scope combinations cover two gene panels, different column names and time units, dense/sparse matrices, and arbitrary matching strata. Its 14 collected cases cover biological split isolation, training-only feature/time fitting, strict data/configuration errors, checkpoint reload, gene reorder/extra/missing behavior, h5ad export, frozen representations and actual transport updates, inference without time labels, complete source support, and the full CLI workflow. Tests use no cohort files, result fixtures, paper paths, or fixed real donor IDs.

Future tests should protect a general user-facing or scientific contract and exercise an independent observable behavior. Add another schema/panel when a change risks data assumptions. Keep research reproduction assertions in the research repository. The small synthetic checks establish execution and contract behavior; they do not establish biological performance on an unseen real cohort.
