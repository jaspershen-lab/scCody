"""Shared-trunk population transport with cell-type heads."""
import math
import torch
from torch import nn

class CellTypeEmbedding(nn.Module):

    def __init__(self, n_cell_types, embed_dim=32):
        super().__init__()
        self.embed = nn.Embedding(n_cell_types, embed_dim)
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, c):
        return self.embed(c)

class DriftHead(nn.Module):
    """Lightweight MLP for cell-type-specific drift deviation."""

    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)

class TimeEncoding(nn.Module):
    """Sinusoidal time encoding."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        if t.dim() == 0:
            t = t.unsqueeze(0)
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)

class CellTypeConditionedFlowMap(nn.Module):
    """Identity-preserving transport with a shared trunk and per-cell-type heads."""

    def __init__(self, base_flow_map, n_cell_types, latent_dim, ct_embed_dim=32, drift_hidden_dim=128, mode='shared_plus_heads'):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_cell_types = n_cell_types
        self.ct_embed_dim = ct_embed_dim
        self.ct_embedding = CellTypeEmbedding(n_cell_types, ct_embed_dim)
        self.time_enc = TimeEncoding(32)
        self.base_flow_map = base_flow_map
        self.input_adapter = nn.Linear(latent_dim + ct_embed_dim, latent_dim)
        self.mode = mode
        if mode in {'shared_plus_heads', 'global_plus_ct_residual'}:
            drift_input_dim = latent_dim + 32
            self.drift_heads = nn.ModuleList([DriftHead(drift_input_dim, latent_dim, drift_hidden_dim) for _ in range(n_cell_types)])
        elif mode == 'concat_embed':
            pass
        else:
            raise ValueError(f'Unknown mode: {mode}')

    def _time_delta(self, s, t, batch_size: int, device, dtype):
        if not torch.is_tensor(s):
            s = torch.tensor(s, device=device, dtype=dtype)
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=device, dtype=dtype)
        if s.ndim == 0:
            s = s.expand(batch_size)
        if t.ndim == 0:
            t = t.expand(batch_size)
        return (t.to(device=device, dtype=dtype) - s.to(device=device, dtype=dtype)).view(batch_size, 1)

    def _compute_drift(self, z, t_scalar, ct):
        """Compute cell-type-specific drift given z, time, and cell type codes."""
        B = z.shape[0]
        if t_scalar.dim() == 0:
            t_scalar = t_scalar.expand(B)
        t_enc = self.time_enc(t_scalar)
        drift_input = torch.cat([z, t_enc], dim=-1)
        drift = torch.zeros_like(z)
        for ct_val in ct.unique():
            mask = ct == ct_val
            if mask.any():
                drift[mask] = self.drift_heads[ct_val.item()](drift_input[mask])
        return drift

    def calc_phi(self, s, t, z, cell_type_idx=None):
        """Return the velocity-like factor in ``F=z+(t-s)*phi``."""
        ct = cell_type_idx
        if ct is None:
            return self.base_flow_map.calc_phi(s, t, z)
        if self.mode == 'global_plus_ct_residual':
            shared_velocity = self.base_flow_map.calc_phi(s, t, z)
        else:
            ct_emb = self.ct_embedding(ct)
            z_adapted = self.input_adapter(torch.cat([z, ct_emb], dim=-1))
            shared_velocity = self.base_flow_map.calc_phi(s, t, z_adapted)
        if self.mode not in {'shared_plus_heads', 'global_plus_ct_residual'}:
            return shared_velocity
        t_scalar = t if isinstance(t, torch.Tensor) else torch.tensor(t, device=z.device, dtype=z.dtype)
        return shared_velocity + self._compute_drift(z, t_scalar, ct)

    def forward(self, s, t, z, cell_type_idx=None):
        """
        Args:
            s: source time scalar or tensor
            t: target time scalar or tensor
            z: (batch, latent_dim) latent state
            cell_type_idx: (batch,) integer cell type indices.
                          If None, uses the shared model.
        Returns:
            z_t: (batch, latent_dim) transported state
        """
        ct = cell_type_idx
        dt = self._time_delta(s, t, z.shape[0], z.device, z.dtype)
        return z + dt * self.calc_phi(s, t, z, ct)
