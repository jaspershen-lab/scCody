import pytest
from sccody import Pipeline
from sccody.demo import synthetic_data


@pytest.fixture(scope="module", params=[(0, "global", "gmvae"), (1, "cell_type", "gmvae_nb"),
                                       (0, "cell_type", "vae"), (1, "global", "vae_nb")])
def fitted(request, tmp_path_factory):
    panel, scope, representation = request.param
    data, config = synthetic_data(panel=panel)
    config.model.scope = scope
    config.model.representation = representation
    config.training.representation_epochs = 1
    config.training.flow_epochs = 2
    config.training.steps_per_epoch = 2
    out = tmp_path_factory.mktemp("pipelines") / "run"
    model = Pipeline(config).fit(data, out)
    target = data[data.obs[config.data.sample_key].isin(config.split.test_samples)].copy()
    source = data[data.obs[config.data.sample_key] == "replicate_0"].copy()
    yield model, data, source, target, out
