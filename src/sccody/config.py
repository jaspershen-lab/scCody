"""Small, strict configuration with explicit data and model contracts."""
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import math
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
    n_genes: int | None = 2000


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
    time_origin: float | None = None
    time_scale: float | None = None


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
    bidirectional: bool = True
    trajectory_unit: str = "sample"
    selection: str = "validation_gene_ot"
    max_reference_cells: int = 512
    seed: int = 42
    device: str = "auto"


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_mapping(cls, mapping):
        if not isinstance(mapping, dict):
            raise ValueError("Configuration must be a mapping")
        types = {"data": DataConfig, "split": SplitConfig, "model": ModelConfig, "training": TrainingConfig}
        unknown = set(mapping) - set(types)
        if unknown:
            raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
        values = {}
        for section, typ in types.items():
            settings = mapping.get(section, {})
            if not isinstance(settings, dict):
                raise ValueError(f"{section} must be a mapping")
            unknown = set(settings) - {f.name for f in fields(typ)}
            if unknown:
                raise ValueError(f"Unknown {section} settings: {sorted(unknown)}")
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
            (m.representation, {"vae", "gmvae", "vae_nb", "gmvae_nb"}, "model.representation"),
            (m.scope, {"global", "cell_type"}, "model.scope"),
            (t.trajectory_unit, {"sample", "time"}, "training.trajectory_unit"),
            (t.selection, {"validation_gene_ot", "last"}, "training.selection"),
        ]:
            if value not in choices:
                raise ValueError(f"{label} must be one of {sorted(choices)}")
        if m.scope == "cell_type" and not d.cell_type_key:
            raise ValueError("cell_type scope requires data.cell_type_key")
        if m.representation.endswith("_nb") and d.input_space != "counts":
            raise ValueError("NB representations require raw counts")
        for name in ("time_key", "sample_key"):
            if not isinstance(getattr(d, name), str) or not getattr(d, name):
                raise ValueError(f"data.{name} must be a nonempty column name")
        if not isinstance(d.stratify_keys, list) or any(not isinstance(k, str) or not k for k in d.stratify_keys):
            raise ValueError("data.stratify_keys must be a list of column names")
        roles = [d.time_key, d.sample_key, *d.stratify_keys, *([d.cell_type_key] if d.cell_type_key else [])]
        if len(set(roles)) != len(roles):
            raise ValueError("Time, sample, cell-type and stratum columns must be distinct")
        if set(roles) & {"_row", "_time", "_sample"}:
            raise ValueError("Observation keys _row, _time and _sample are reserved for internal preparation")
        for name in ("validation_samples", "test_samples"):
            if not isinstance(getattr(s, name), list) or any(not isinstance(x, str) for x in getattr(s, name)):
                raise ValueError(f"split.{name} must be a list of sample ID strings")
        if set(s.validation_samples) & set(s.test_samples):
            raise ValueError("Validation and test sample IDs overlap")
        for name in ("validation_fraction", "test_fraction"):
            value = getattr(s, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"split.{name} must be in [0, 1)")
        if s.validation_fraction + s.test_fraction >= 1:
            raise ValueError("Split fractions must sum to less than 1")
        for name, value in {
            "latent_dim": m.latent_dim, "clusters": m.clusters,
            "representation_epochs": t.representation_epochs,
            "representation_batch_size": t.representation_batch_size,
            "flow_epochs": t.flow_epochs, "steps_per_epoch": t.steps_per_epoch,
            "cell_batch_size": t.cell_batch_size, "max_reference_cells": t.max_reference_cells,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if d.n_genes is not None and (not isinstance(d.n_genes, int) or isinstance(d.n_genes, bool) or d.n_genes < 1):
            raise ValueError("data.n_genes must be null or a positive integer")
        for dims in (m.hidden_dims, m.flow_hidden_dims):
            if not isinstance(dims, list) or any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in dims):
                raise ValueError("Hidden dimensions must be lists of positive integers")
        for name, value in {"normalize_total": d.normalize_total, "representation_lr": t.representation_lr,
                            "flow_lr": t.flow_lr, "sinkhorn_blur": t.sinkhorn_blur}.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("beta", "gene_ot_weight", "latent_ot_weight", "endpoint_weight"):
            value = getattr(t, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"training.{name} must be finite and nonnegative")
        if t.gene_ot_weight + t.latent_ot_weight == 0:
            raise ValueError("At least one transport loss weight must be positive")
        if not 0 <= m.dropout < 1 or not 0 < t.sinkhorn_scaling < 1:
            raise ValueError("dropout must be in [0,1); sinkhorn_scaling must be in (0,1)")
        if (m.time_origin is None) != (m.time_scale is None):
            raise ValueError("Set time_origin and time_scale together, or leave both null")
        if m.time_scale is not None and (not math.isfinite(m.time_scale) or m.time_scale <= 0 or not math.isfinite(m.time_origin)):
            raise ValueError("Time coordinates must be finite with a positive scale")
        if not isinstance(t.seed, int) or t.seed < 0:
            raise ValueError("training.seed must be a nonnegative integer")
        if not isinstance(t.device, str):
            raise ValueError("training.device must be a torch device string or auto")
