"""Data, model and training settings for population transport."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import yaml


@dataclass
class DataConfig:
    time_key: str = "time"
    sample_key: str = "sample"
    cell_type_key: str | None = None
    stratify_keys: list[str] = field(default_factory=list)
    layer: str | None = None
    input_space: str = "counts"
    normalize_total: float = 10000.0
    n_genes: int | None = None
    max_train_time: float | None = None


@dataclass
class SplitConfig:
    validation_samples: list[str] = field(default_factory=list)
    test_samples: list[str] = field(default_factory=list)
    validation_fraction: float = 0.15
    test_fraction: float = 0.15


@dataclass
class ModelConfig:
    representation: str = "gmvae"
    latent_dim: int = 50
    hidden_dims: list[int] = field(default_factory=lambda: [512, 512])
    clusters: int = 20
    dropout: float = 0.05
    scope: str = "global"
    flow_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    time_origin: float | None = 0.0
    time_scale: float | None = 100.0
    no_softplus_decode: bool = False
    cell_types: list[str] = field(default_factory=list)


@dataclass
class TrainingConfig:
    representation_epochs: int = 100
    representation_lr: float = 0.001
    representation_batch_size: int = 512
    beta: float = 0.0001
    flow_epochs: int = 30
    steps_per_epoch: int = 100
    cell_batch_size: int = 384
    flow_lr: float = 0.0005
    gene_ot_weight: float = 1.0
    latent_ot_weight: float = 1.0
    endpoint_weight: float = 1.0
    sinkhorn_blur: float = 0.05
    sinkhorn_scaling: float = 0.5
    trajectory_unit: str = "sample"
    selection: str = "validation_gene_ot"
    max_reference_cells: int = 512
    seed: int = 42
    device: str = "auto"
    trajectory_nodes: int = 24
    sources_per_target: int = 6
    endpoint_pairs: int = 8
    min_population_cells: int = 1
    min_population_nodes: int = 2
    adam_betas: list[float] = field(default_factory=lambda: [0.95, 0.99])
    representation_checkpoint: str | None = None


@dataclass
class ScCodyConfig:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_mapping(cls, mapping):
        types = {"data": DataConfig, "split": SplitConfig, "model": ModelConfig, "training": TrainingConfig}
        unknown = set(mapping) - set(types)
        if unknown:
            raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
        values = {}
        for section, typ in types.items():
            settings = mapping.get(section, {})
            if section == "model" and settings.get("ct_flow_mode") == "shared_plus_heads":
                settings = {k: v for k, v in settings.items() if k != "ct_flow_mode"}
            if section == "training":
                # Accept obsolete settings recorded in existing configs and checkpoints.
                settings = {k: v for k, v in settings.items() if k not in {
                    "pair_policy", "bidirectional", "reference_times", "reference_include_untrained_types", "checkpoint_every_epochs"}}
            values[section] = typ(**settings)
        config = cls(**values)
        config.validate()
        return config

    @classmethod
    def load(cls, path):
        return cls.from_mapping(yaml.safe_load(Path(path).read_text()))

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    def validate(self):
        d, s, m, t = self.data, self.split, self.model, self.training
        for value, choices, label in [
            (d.input_space, {"counts", "log1p"}, "data.input_space"),
            (m.scope, {"global", "cell_type"}, "model.scope"),
            (t.trajectory_unit, {"sample", "time"}, "training.trajectory_unit"),
            (t.selection, {"validation_gene_ot", "last"}, "training.selection"),
        ]:
            if value not in choices:
                raise ValueError(f"{label} must be one of {sorted(choices)}")
        if m.scope == "cell_type" and not d.cell_type_key:
            raise ValueError("cell_type scope requires data.cell_type_key")
        roles = [d.time_key, d.sample_key, *d.stratify_keys, *([d.cell_type_key] if d.cell_type_key else [])]
        if len(set(roles)) != len(roles):
            raise ValueError("Time, sample, cell-type and stratum columns must be distinct")
        if set(roles) & {"_row", "_time", "_sample"}:
            raise ValueError("Observation keys _row, _time and _sample are reserved for internal preparation")
        if set(s.validation_samples) & set(s.test_samples):
            raise ValueError("Validation and test sample IDs overlap")
        if len(set(m.cell_types)) != len(m.cell_types):
            raise ValueError("model.cell_types must be distinct")
        if (m.time_origin is None) != (m.time_scale is None):
            raise ValueError("Set time_origin and time_scale together, or leave both null")
