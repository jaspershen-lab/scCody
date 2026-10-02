import anndata as ad
import pytest
from sccody.cli import main
from sccody.demo import synthetic_data


def test_complete_cli_workflow_on_alternate_schema(tmp_path):
    data, config = synthetic_data(panel=1)
    config.model.scope = "cell_type"
    config.training.representation_epochs = 1
    config.training.flow_epochs = 1
    config.training.steps_per_epoch = 1
    data.write_h5ad(tmp_path / "input.h5ad")
    config.save(tmp_path / "config.yaml")
    common = ["--data", str(tmp_path / "input.h5ad"), "--config", str(tmp_path / "config.yaml")]
    assert main(["validate", *common]) == 0
    run = tmp_path / "run"
    assert main(["run", *common, "--outdir", str(run), "--candidate-times", "0,12,24"]) == 0
    assert (run / "evaluation" / "metrics.json").exists()
    assert (run / "evaluation" / "reconstruction.csv").exists()
    assert main(["train", *common, "--outdir", str(run)]) == 2
    unlabeled = data[data.obs[config.data.sample_key] == "replicate_10"].copy()
    del unlabeled.obs[config.data.time_key]
    unlabeled.write_h5ad(tmp_path / "unlabelled.h5ad")
    assert main(["localize", "--model", str(run), "--data", str(tmp_path / "unlabelled.h5ad"),
                 "--outdir", str(tmp_path / "localization"), "--candidate-times", "0,12,24", "--max-cells", "8"]) == 0


def test_cli_help_and_configuration_errors(tmp_path, capsys):
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0
    assert "localize" in capsys.readouterr().out
    path = str(tmp_path / "template.yaml")
    assert main(["init-config", path]) == 0
    assert main(["init-config", path]) == 2
