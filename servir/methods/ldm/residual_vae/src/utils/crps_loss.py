# src/utils/crps_loss_clean.py
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class CRPSLoss(nn.Module):
    """
    Empirical CRPS Loss for ensemble predictions from VAE.
    
    CRPS Formula: CRPS = (1/N) Σ|x_i - y| - (1/2N²) ΣΣ|x_i - x_j|
    
    Where:
    - x_i are ensemble predictions (N samples)
    - y is the observation (ground truth)
    - First term: Mean absolute error between predictions and truth
    - Second term: Mean pairwise absolute differences between predictions (diversity penalty)
    """
    
    def __init__(self, num_samples=20, reduction='mean'):
        """
        Initialize CRPS loss.
        
        Args:
            num_samples: Number of ensemble samples to generate
            reduction: How to reduce the loss ('mean', 'sum', 'none')
        """
        super().__init__()
        self.num_samples = num_samples
        self.reduction = reduction
    
    def empirical_crps_vectorized(self, forecasts, observations):
        """
        Vectorized CRPS computation using the efficient formula:
        CRPS = (1/N) Σ|x_i - y| - (1/2N²) ΣΣ|x_i - x_j|
        
        Args:
            forecasts: (N_samples, batch, channels, depth, height, width)
            observations: (batch, channels, depth, height, width)
        
        Returns:
            crps: (batch, channels, depth, height, width)
        """
        # Expand observations for broadcasting
        obs_expanded = observations.unsqueeze(0)  # (1, batch, channels, depth, height, width)
        
        # First term: (1/N) Σ|x_i - y|
        # Mean absolute error between each prediction and observation
        first_term = torch.mean(torch.abs(forecasts - obs_expanded), dim=0)
        
        # Second term: (1/2N²) ΣΣ|x_i - x_j|
        # Mean pairwise absolute differences between predictions
        forecasts_i = forecasts.unsqueeze(1)  # (N, 1, B, C, D, H, W)
        forecasts_j = forecasts.unsqueeze(0)  # (1, N, B, C, D, H, W)
        pairwise_diff = torch.abs(forecasts_i - forecasts_j)  # (N, N, B, C, D, H, W)
        second_term = torch.mean(pairwise_diff, dim=(0, 1)) / 2.0  # Average and divide by 2
        
        # CRPS = accuracy - diversity penalty
        crps = first_term - second_term
        
        return crps
    
    def forward(self, mu, logvar, decoder, target):
        """
        Compute CRPS loss by generating ensemble from VAE latent space.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            decoder: Decoder function/module
            target: Ground truth
        
        Returns:
            crps_loss: Scalar loss value
        """
        # Generate ensemble predictions
        B, C, D, H, W = target.shape
        ensemble = torch.empty((self.num_samples, B, C, D, H, W), 
                              device=target.device, dtype=target.dtype)
        
        for i in range(self.num_samples):
            # Sample from latent space
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            
            # Decode to get prediction
            pred = decoder(z)
            ensemble[i] = pred
        
        # Compute CRPS
        crps = self.empirical_crps_vectorized(ensemble, target)
        
        # Apply reduction
        if self.reduction == 'mean':
            return crps.mean()
        elif self.reduction == 'sum':
            return crps.sum()
        else:
            return crps
    
    def evaluate(self, mu, logvar, decoder, target):
        """
        Evaluate CRPS loss (alias for forward method for validation).
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            decoder: Decoder function/module
            target: Ground truth
        
        Returns:
            crps_loss: Scalar loss value
        """
        return self.forward(mu, logvar, decoder, target)