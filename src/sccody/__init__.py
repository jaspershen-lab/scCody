"""User-facing single-cell population transport."""
from .config import ScCodyConfig

__version__ = "0.1.0"
__all__ = ["ScCodyConfig", "ScCody", "LocalizationResult", "localize_ensemble",
           "decompose_disease", "decompose_displacements", "decode_components"]


def __getattr__(name):
    if name in {"ScCody", "LocalizationResult"}:
        from .model import ScCody, LocalizationResult
        return {"ScCody": ScCody, "LocalizationResult": LocalizationResult}[name]
    if name == "localize_ensemble":
        from .projection import localize_ensemble
        return localize_ensemble
    if name in {"decompose_disease", "decompose_displacements", "decode_components"}:
        from . import decomposition
        return getattr(decomposition, name)
    raise AttributeError(name)
