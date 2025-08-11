# src/utils/crps_loss.py
import torch
import torch.nn as nn
import math
import logging

logger = logging.getLogger(__name__)

class CRPSLoss(nn.Module):
    """
    Empirical CRPS Loss for ensemble predictions from VAE.
    """
    
    def __init__(self, num_samples=20, reduction='mean', debug=False):
        """
        Initialize CRPS loss.
        
        Args:
            num_samples: Number of ensemble samples to generate
            reduction: How to reduce the loss ('mean', 'sum', 'none')
            debug: Whether to print debug information
        """
        super().__init__()
        self.num_samples = num_samples
        self.reduction = reduction
        self.debug = debug
        
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for VAE sampling with device safety.
        
        Args:
            mu: Mean tensor
            logvar: Log variance tensor
            
        Returns:
            Sampled tensor on the same device as input
        """
        std = torch.exp(0.5 * logvar)
        # Ensure eps is on the same device as mu/logvar
        eps = torch.randn_like(std, device=std.device)
        return mu + eps * std
    
    def empirical_crps_vectorized(self, forecasts, observations):
        """
        Vectorized CRPS computation using the efficient formula:
        CRPS ≈ (1/N) Σ|x_i - y| - (1/2N²) ΣΣ|x_i - x_j|
        
        Args:
            forecasts: (N_samples, batch, channels, depth, height, width)
            observations: (batch, channels, depth, height, width)
        
        Returns:
            crps: (batch, channels, depth, height, width)
        """
        N_samples = forecasts.shape[0]
        
        # Expand observations to match forecasts shape for broadcasting
        obs_expanded = observations.unsqueeze(0)  # (1, batch, channels, depth, height, width)
        
        # First term: (1/N) Σ|x_i - y|
        # Compute absolute differences between each forecast and observation
        abs_diff_obs = torch.abs(forecasts - obs_expanded)  # (N_samples, batch, channels, depth, height, width)
        first_term = torch.mean(abs_diff_obs, dim=0)  # Average over samples: (batch, channels, depth, height, width)
        
        # Second term: (1/2N²) ΣΣ|x_i - x_j|
        # Compute pairwise absolute differences between forecasts
        # forecasts: (N, B, C, D, H, W)
        # We need to compute |x_i - x_j| for all i,j pairs
        
        # Expand forecasts for pairwise computation
        forecasts_i = forecasts.unsqueeze(1)  # (N, 1, B, C, D, H, W)
        forecasts_j = forecasts.unsqueeze(0)  # (1, N, B, C, D, H, W)
        
        # Compute absolute pairwise differences
        pairwise_diff = torch.abs(forecasts_i - forecasts_j)  # (N, N, B, C, D, H, W)
        
        # Average over both sample dimensions
        second_term = torch.mean(pairwise_diff, dim=(0, 1))  # (B, C, D, H, W)
        second_term = second_term / 2.0  # Divide by 2 as per formula
        
        # Combine terms: CRPS = first_term - second_term
        crps = first_term - second_term
        
        # Debug logging
        if self.debug:
            logger.info(f"CRPS first_term mean: {first_term.mean().item():.6f}")
            logger.info(f"CRPS second_term mean: {second_term.mean().item():.6f}")
            logger.info(f"CRPS final mean: {crps.mean().item():.6f}")
        
        return crps
    
    def forward(self, mu, logvar, decoder, target):
        """
        Compute CRPS loss by generating ensemble from VAE latent space.
        
        Args:
            mu: Mean of latent distribution (B, latent_channels, D, H, W)
            logvar: Log variance of latent distribution (B, latent_channels, D, H, W)
            decoder: Decoder function/module
            target: Ground truth (B, C, D, H, W)
        
        Returns:
            crps_loss: Scalar loss value
        """
        # Pre-allocate ensemble tensor for efficiency
        B, C, D, H, W = target.shape
        ensemble = torch.empty((self.num_samples, B, C, D, H, W), 
                              device=target.device, dtype=target.dtype)
        
        # Generate ensemble predictions with device safety
        for i in range(self.num_samples):
            # Sample from latent space
            z = self.reparameterize(mu, logvar)
            # Decode to get prediction (preserve gradients for training)
            pred = decoder(z)
            ensemble[i] = pred
        
        # Compute CRPS
        crps = self.empirical_crps_vectorized(ensemble, target)
        
        # Apply reduction
        if self.reduction == 'mean':
            return crps.mean()
        elif self.reduction == 'sum':
            return crps.sum()
        elif self.reduction == 'none':
            return crps
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")
    
    def forward_efficient(self, mu, logvar, decoder, target, training=True):
        """
        More efficient version with optional gradient management.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution  
            decoder: Decoder function/module
            target: Ground truth
            training: Whether in training mode (affects gradient computation)
        
        Returns:
            crps_loss: Scalar loss value
        """
        B, C, D, H, W = target.shape
        ensemble = torch.empty((self.num_samples, B, C, D, H, W), 
                              device=target.device, dtype=target.dtype)
        
        for i in range(self.num_samples):
            z = self.reparameterize(mu, logvar)
            pred = decoder(z)
            # Optional: detach for inference to save memory
            ensemble[i] = pred if training else pred.detach()
        
        crps = self.empirical_crps_vectorized(ensemble, target)
        
        if self.reduction == 'mean':
            return crps.mean()
        elif self.reduction == 'sum':
            return crps.sum()
        elif self.reduction == 'none':
            return crps
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")
    
    @torch.no_grad()
    def evaluate(self, mu, logvar, decoder, target):
        """
        Evaluation-only CRPS computation without gradient tracking.
        Use this for validation/testing to save memory.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution  
            decoder: Decoder function/module
            target: Ground truth
        
        Returns:
            crps_loss: Scalar loss value (no gradients)
        """
        B, C, D, H, W = target.shape
        ensemble = torch.empty((self.num_samples, B, C, D, H, W), 
                              device=target.device, dtype=target.dtype)
        
        for i in range(self.num_samples):
            z = self.reparameterize(mu, logvar)
            pred = decoder(z)
            ensemble[i] = pred
        
        crps = self.empirical_crps_vectorized(ensemble, target)
        
        if self.reduction == 'mean':
            return crps.mean()
        elif self.reduction == 'sum':
            return crps.sum()
        elif self.reduction == 'none':
            return crps
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")