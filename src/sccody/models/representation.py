"""Reusable representation models extracted from scCodyFlow's research implementation."""
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dist

ACT_FUNC_MAP = {"relu": nn.ReLU(), "gelu": nn.GELU(), "softplus": nn.Softplus()}

def sample_gaussian(mean, std):
    return mean + std.clamp_min(1e-8) * torch.randn_like(mean)

def negative_binomial_loss(x, mu, theta, eps=1e-8, is_log_input=True):
    """
    Computes NB negative log-likelihood.
    Robust to log-transformed inputs by inverting them to counts.
    """
    if is_log_input:
        x = torch.expm1(x) # Invert log1p -> counts
        x = F.relu(x)      # Safety

    def log_gamma(v): return torch.lgamma(v)

    t1 = log_gamma(x + theta + eps) - log_gamma(theta + eps) - log_gamma(x + 1.0)
    t2 = theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
    t3 = x * (torch.log(mu + eps) - torch.log(theta + mu + eps))

    return -torch.mean(t1 + t2 + t3)

class MLP(nn.Sequential):
    """
    Helper to build a simple Multi-Layer Perceptron.
    """
    def __init__(self, in_dim, hidden_list, out_dim, act_name="relu", dropout=0.1):
        layers = []
        act_func = ACT_FUNC_MAP[act_name]
        last_dim = in_dim
        for h in hidden_list:
            layers.append(nn.Linear(last_dim, h))
            #layers.append(nn.BatchNorm1d(h))
            layers.append(act_func)
            if dropout > 0: layers.append(nn.Dropout(dropout))
            last_dim = h
        layers.append(nn.Linear(last_dim, out_dim))
        super().__init__(*layers)

class VAE(nn.Module):
    """
    Parent class for all VAEs.
    Implements standard Gaussian reparameterization and MLP construction.
    """
    def __init__(
        self,
        input_dim,
        latent_dim=16,
        hidden_dims=(128, 64),
        act="relu",
        dropout=0.1,
        decoder_hidden_dims=None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.uses_library_size = False # Default: False

        hidden_dims = tuple(hidden_dims or ())
        if decoder_hidden_dims is None:
            decoder_hidden_dims = tuple(reversed(hidden_dims))
        else:
            decoder_hidden_dims = tuple(decoder_hidden_dims or ())

        # Standard Encoder: x -> z_mu, z_logvar
        self.encoder = MLP(input_dim, hidden_dims, latent_dim * 2, act, dropout)

        # Standard Decoder: z -> x_hat (can be overridden by subclasses)
        self.decoder = MLP(latent_dim, decoder_hidden_dims, input_dim, act, dropout)

    def encode(self, x):
        """Standard Gaussian Encoder"""
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Standard Reparameterization Trick"""
        std = torch.exp(0.5 * logvar) + 1e-4
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Standard Decoder (ignores library_size).
        Returns: tensor (reconstruction)
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Unified forward pass.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss(self, x, recon, mu, logvar, beta_kl=1.0):
        """
        Default Loss: MSE + Standard KL.
        """
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + beta_kl * kl_loss

        logs = {
            "loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item()
        }
        return total_loss, logs, recon_loss, kl_loss

    def sample(self, x, use_mean: bool = False):
        """
        Default sampling: diagonal q(z|x) using encode() output.
        Returns z that dynamics can evolve.
        """
        mu, logvar = self.encode(x)
        if use_mean:
            return mu

        std = torch.exp(0.5 * logvar) + 1e-4
        z = sample_gaussian(mu, std)
        return z

class GMVAE(VAE):
    """
    Gaussian Mixture VAE.
    - Adds GMM Prior parameters.
    - Overrides loss() to use GMM-KL.
    """
    def __init__(
        self,
        n_input,
        latent_dim,
        hidden_dims=(128, 64),
        act="relu",
        dropout=0.1,
        n_clusters=20,
        decoder_hidden_dims=None,
    ):
        super().__init__(
            n_input,
            latent_dim,
            hidden_dims,
            act,
            dropout,
            decoder_hidden_dims=decoder_hidden_dims,
        )

        self.n_clusters = n_clusters

        # Learnable GMM Prior
        self.prior_mu = nn.Parameter(torch.randn(n_clusters, latent_dim))
        self.prior_logvar = nn.Parameter(torch.zeros(n_clusters, latent_dim))
        self.prior_pi = nn.Parameter(torch.ones(n_clusters))

    def get_gmm_prior(self):
        mix = dist.Categorical(logits=self.prior_pi)
        comp = dist.Independent(
            dist.Normal(self.prior_mu, (self.prior_logvar.exp() + 1e-4).sqrt()), 1
        )
        return dist.MixtureSameFamily(mix, comp)

    def loss(self, x, recon, mu, logvar, beta_kl=1.0):
        # 1. Reconstruction (MSE)
        recon_loss = F.mse_loss(recon, x, reduction='mean')

        # 2. GMM KL Divergence (Monte Carlo Approx)
        # Sample z from q(z|x)
        z = self.reparameterize(mu, logvar)

        # log q(z|x)
        q_dist = dist.Normal(mu, (logvar.exp() + 1e-4).sqrt())
        log_q = q_dist.log_prob(z).sum(-1)

        # log p(z)
        p_dist = self.get_gmm_prior()
        log_p = p_dist.log_prob(z)

        # KL = log q - log p
        kl_loss = (log_q - log_p).mean()

        total_loss = recon_loss + beta_kl * kl_loss

        logs = {
            "loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item()
        }
        return total_loss, logs, recon_loss, kl_loss

class CountEncoderMixin:
    """Encode raw counts through log1p while keeping NB reconstruction count-native."""

    def encode(self, x):
        x_enc = torch.log1p(torch.clamp(x, min=0.0))
        return super().encode(x_enc)

class VAE_NB(CountEncoderMixin, VAE):
    """Count-native NB VAE with the same latent geometry as ``VAE``.

    The decoder returns raw logits. ``loss`` applies exactly one softplus to get
    the positive NB mean, avoiding the old double-softplus/SCVI tuple path.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.px_r = nn.Parameter(torch.zeros(self.input_dim))

    def loss(self, x, recon, mu, logvar, beta_kl=1.0):
        x_counts = torch.clamp(x, min=0.0)
        mu_counts = F.softplus(recon)
        theta = torch.exp(self.px_r) + 1e-4
        recon_loss = negative_binomial_loss(x_counts, mu_counts, theta, is_log_input=False)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + beta_kl * kl_loss
        logs = {
            "loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
        }
        return total_loss, logs, recon_loss, kl_loss

class GMVAE_NB(CountEncoderMixin, GMVAE):
    """Count-native NB GMVAE with the same GMM prior as ``GMVAE``."""

    def __init__(self, input_dim=None, *args, n_input=None, **kwargs):
        n_input = input_dim if n_input is None else n_input
        if n_input is None:
            raise TypeError("GMVAE_NB requires input_dim or n_input")
        super().__init__(n_input, *args, **kwargs)
        self.px_r = nn.Parameter(torch.zeros(self.input_dim))

    def loss(self, x, recon, mu, logvar, beta_kl=1.0):
        x_counts = torch.clamp(x, min=0.0)
        mu_counts = F.softplus(recon)
        theta = torch.exp(self.px_r) + 1e-4
        recon_loss = negative_binomial_loss(x_counts, mu_counts, theta, is_log_input=False)

        z = self.reparameterize(mu, logvar)
        q_dist = dist.Normal(mu, (logvar.exp() + 1e-4).sqrt())
        log_q = q_dist.log_prob(z).sum(-1)
        log_p = self.get_gmm_prior().log_prob(z)
        kl_loss = (log_q - log_p).mean()

        total_loss = recon_loss + beta_kl * kl_loss
        logs = {
            "loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
        }
        return total_loss, logs, recon_loss, kl_loss
