from __future__ import annotations

import torch
import torch.nn as nn

from .representation import ACT_FUNC_MAP


class FlowMapTimeEmbed(nn.Module):
    """FlowMap MLP X(s, t, z) = z + (t - s) * phi(s, t, z)."""

    def __init__(self, latent_dim: int, hidden_sizes=None, act_name: str = "relu", use_residual: bool = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_residual = use_residual
        hidden_sizes = [256, 256] if hidden_sizes is None else list(hidden_sizes)
        act = ACT_FUNC_MAP.get(act_name, nn.GELU())

        layers: list[nn.Module] = []
        last_dim = latent_dim + 2
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(act)
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, latent_dim))
        self.phi_net = nn.Sequential(*layers)
        nn.init.zeros_(self.phi_net[-1].weight)
        nn.init.zeros_(self.phi_net[-1].bias)

    def _prepare_time_pair(self, s, t, batch_size: int, device, dtype):
        if not torch.is_tensor(s):
            s = torch.tensor(s, device=device, dtype=dtype)
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=device, dtype=dtype)
        if s.ndim == 0:
            s = s.expand(batch_size)
        if t.ndim == 0:
            t = t.expand(batch_size)
        return (
            s.to(device=device, dtype=dtype).view(batch_size, 1),
            t.to(device=device, dtype=dtype).view(batch_size, 1),
        )

    def calc_phi(self, s, t, z):
        batch_size, _latent_dim = z.shape
        s_bt, t_bt = self._prepare_time_pair(s, t, batch_size, z.device, z.dtype)
        return self.phi_net(torch.cat([z, s_bt, t_bt], dim=1))

    def calc_b(self, t, x):
        return self.calc_phi(t, t, x)

    def calc_weight(self, s, t):
        if torch.is_tensor(s):
            device = s.device
            dtype = s.dtype
        elif torch.is_tensor(t):
            device = t.device
            dtype = t.dtype
        else:
            device = self.phi_net[0].weight.device
            dtype = self.phi_net[0].weight.dtype
        return torch.tensor(1.0, device=device, dtype=dtype)

    def forward(self, s, t, z):
        batch_size, _latent_dim = z.shape
        s_bt, t_bt = self._prepare_time_pair(s, t, batch_size, z.device, z.dtype)
        return z + (t_bt - s_bt) * self.calc_phi(s_bt.squeeze(-1), t_bt.squeeze(-1), z)
