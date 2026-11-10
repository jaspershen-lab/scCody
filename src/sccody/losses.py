"""Population Sinkhorn divergence and the research barycentric coupling."""
from functools import lru_cache
import torch
from geomloss import SamplesLoss


@lru_cache(maxsize=16)
def _solver(blur, scaling, potentials=False):
    return SamplesLoss("sinkhorn", p=2, blur=blur, scaling=scaling,
                       debias=not potentials, potentials=potentials, backend="tensorized")


def sinkhorn_divergence(source, target, *, blur=0.05, scaling=0.5):
    """Debiased uniform p=2 divergence; GeomLoss uses half squared distance."""
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1] or not len(source) or not len(target):
        raise ValueError("Sinkhorn requires nonempty populations with aligned features")
    # Analytic point-mass limit avoids GeomLoss's zero-diameter epsilon schedule.
    with torch.no_grad():
        constant = (torch.equal(source.amin(0), source.amax(0)) and
                    torch.equal(target.amin(0), target.amax(0)))
    if constant:
        return 0.5 * (source.mean(0) - target.mean(0)).square().sum()
    return _solver(float(blur), float(scaling))(source, target)


def sinkhorn_plan(source, target, *, blur=0.05, scaling=0.5):
    """Research conditional coupling, with the uniform source marginal enforced.

    Raw (non-debiased) target potentials define P(j|i); this is the coupling
    used for barycentric fields, not a debiased scalar divergence. Finite
    solver tolerance can leave a target-marginal residual.
    """
    if not len(source) or not len(target):
        return source.new_zeros((len(source), len(target)))
    _, potential = _solver(float(blur), float(scaling), True)(source, target)
    cost = torch.cdist(source, target).square().mul(0.5)
    conditional = torch.softmax((potential.reshape(1, -1) - cost) / float(blur)**2, dim=1)
    if not torch.isfinite(conditional).all():
        raise FloatingPointError("Nonfinite Sinkhorn conditional coupling")
    return conditional / len(source)
