import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from sccody import Pipeline


def test_training_checkpoint_roundtrip_and_gene_alignment(fitted, tmp_path):
    model, data, source, target, out = fitted
    reloaded = Pipeline.load(out, device="cpu")
    time = float(data.obs[model.config.data.time_key].max())
    generated = model.generate(source, time)
    restored = reloaded.generate(source, time)
    np.testing.assert_array_equal(generated.X, restored.X)
    assert generated.obs_names.equals(source.obs_names)
    assert generated.var_names.tolist() == model.genes
    assert np.isfinite(generated.X).all() and (generated.X >= 0).all()
    shuffled = source[:, np.arange(source.n_vars)[::-1]].copy()
    np.testing.assert_allclose(generated.X, reloaded.generate(shuffled, time).X, rtol=1e-6, atol=1e-6)
    dense = source.X.toarray() if hasattr(source.X, "toarray") else source.X
    extra = ad.AnnData(np.c_[dense, np.full(source.n_obs, 10000, dtype=np.float32)], obs=source.obs.copy(),
                      var=pd.DataFrame(index=pd.Index([*source.var_names, "extra_gene"], dtype=object)))
    np.testing.assert_array_equal(generated.X, reloaded.generate(extra, time).X)
    generated.write_h5ad(tmp_path / "generated.h5ad")
    assert ad.read_h5ad(tmp_path / "generated.h5ad").shape == generated.shape
    with pytest.raises(ValueError, match="missing"):
        reloaded.generate(source[:, source.var_names != model.genes[0]].copy(), time)
    with pytest.raises(FileExistsError):
        Pipeline(model.config).fit(data, out)
    assert 1 <= reloaded.selection["epoch"] <= model.config.training.flow_epochs
    assert reloaded.selection["test_samples_used"] is False
    history = pd.read_csv(out / "flow_history.csv")
    assert history["validation_gene_ot"].notna().all()
    assert history["validation_gene_ot"].nunique() > 1
    assert model.selection["epoch"] == int(history.loc[history["validation_gene_ot"].idxmin(), "epoch"])
    ref_ids = {cell for ref in model.reference for cell in ref["cell_ids"]}
    heldout = set(data.obs_names[data.obs[model.config.data.sample_key].isin(
        model.config.split.validation_samples + model.config.split.test_samples)])
    assert not ref_ids.intersection(heldout)
    assert all(not p.requires_grad for p in model.representation.parameters())
    transport = model.transport.base_flow_map if model.config.model.scope == "cell_type" else model.transport
    assert torch.count_nonzero(transport.phi_net[-1].weight).item() > 0


def test_localization_does_not_read_true_time_and_requires_complete_sources(fitted):
    model, data, source, target, _ = fitted
    time_key = model.config.data.time_key
    grid = sorted(data.obs[time_key].unique())
    original = model.localize(target, grid, max_cells=8)
    assert (original.predictions["status"] == "ok").all()
    changed = target.copy()
    changed.obs[time_key] = -9876
    pd.testing.assert_frame_equal(original.predictions, model.localize(changed, grid, max_cells=8).predictions)
    del changed.obs[time_key]
    unlabelled = model.localize(changed, grid, max_cells=8)
    pd.testing.assert_frame_equal(original.curves, unlabelled.curves)
    unsupported = model.localize(changed, grid, source_times=[-12345], max_cells=8)
    assert (unsupported.predictions["status"] == "unsupported_source").all()
    assert unsupported.predictions["predicted_time"].isna().all()
    assert unsupported.curves.empty
    changed.obs["plate"] = "unseen_plate"
    assert (model.localize(changed, grid, max_cells=8).predictions["status"] == "unsupported_source").all()
