import numpy as np
import pytest
from sccody import PipelineConfig
from sccody.data import prepare, validate_data
from sccody.demo import synthetic_data


@pytest.mark.parametrize("panel", [0, 1])
def test_holdout_expression_and_time_cannot_change_fitted_panel(panel):
    data, config = synthetic_data(panel=panel)
    first = prepare(data, config)
    altered = data.copy()
    altered.X = altered.X.toarray() if hasattr(altered.X, "toarray") else altered.X.copy()
    rows = np.r_[first.validation_indices, first.test_indices]
    altered.X[rows] = 100000
    altered.obs.loc[altered.obs.index[rows], config.data.time_key] = 100000
    second = prepare(altered, config)
    assert first.genes == second.genes
    assert (first.time_origin, first.time_scale) == (second.time_origin, second.time_scale)
    train = set(first.split_table.loc[first.split_table["split"] == "train", config.data.sample_key])
    assert not train.intersection(config.split.validation_samples + config.split.test_samples)


def test_actionable_validation_and_strict_configuration():
    data, config = synthetic_data()
    with pytest.raises(ValueError, match="Unknown data settings"):
        PipelineConfig.from_mapping({"data": {"time_keey": "hour"}})
    data.obs.loc[data.obs.index[0], "time"] = 7
    with pytest.raises(ValueError, match="one 'time'"):
        prepare(data, config)
    data, config = synthetic_data()
    del data.obs["sample"]
    with pytest.raises(ValueError, match="Missing observation column"):
        validate_data(data, config)
    config.model.representation = "vae_nb"
    config.data.input_space = "log1p"
    with pytest.raises(ValueError, match="raw counts"):
        config.validate()


def test_processed_input_and_fractional_counts_are_distinct():
    data, config = synthetic_data()
    data.X = np.log1p(data.X)
    with pytest.raises(ValueError, match="integer-valued"):
        prepare(data, config)
    config.data.input_space = "log1p"
    assert len(prepare(data, config).genes) == 10
