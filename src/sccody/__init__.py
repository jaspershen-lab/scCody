"""User-facing single-cell population transport."""
from .config import PipelineConfig

__version__ = "0.1.0"
__all__ = ["PipelineConfig", "Pipeline", "LocalizationResult"]


def __getattr__(name):
    if name in {"Pipeline", "LocalizationResult"}:
        from .pipeline import Pipeline, LocalizationResult
        return {"Pipeline": Pipeline, "LocalizationResult": LocalizationResult}[name]
    raise AttributeError(name)
