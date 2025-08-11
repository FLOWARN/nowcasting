# vae_models/vae_difference_generative.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SimpleVAE3D_DifferenceGenerative(nn.Module):
    """
    3D Variational Autoencoder for difference encoding with generative fine-tuning capability.
    
    This model is designed to work with precipitation differences (GT - Langragian predictions)
    and can be fine-tuned for generative sampling using CRPS loss.
    """
    
    def __init__(self, input_channels: int = 1, latent_dim: int = 32):
        """
        Initialize the 3D VAE model for difference encoding.
        
        Args:
            input_channels: Number of input channels
            latent_dim: Dimension of latent space
        """
        super().__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        
        # Build encoder and decoder
        self._build_encoder()
        self._build_decoder()
        
        logger.info(f"Initialized SimpleVAE3D_DifferenceGenerative with input_channels={input_channels}, latent_dim={latent_dim}")

    def _build_encoder(self) -> None:
        """Construct the encoder network."""
        # Encoder layers optimized for difference data
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv3d(self.input_channels, 64, kernel_size=3, padding=1, stride=(1,1,2)),
            nn.GroupNorm(32, 64),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=1),
            nn.GroupNorm(32, 64),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=1),
            nn.SELU(),
            
            # Block 2
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 0), stride=1),
            nn.GroupNorm(32, 64),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=1),
            nn.GroupNorm(32, 64),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=1),
            nn.SELU(),
            
            # Block 3
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=2),
            nn.GroupNorm(32, 64),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=1),
            
            # Block 4
            nn.Conv3d(64, 64, kernel_size=(1, 3, 1), stride=(1, 2, 1), padding=(0, 1, 0)),
            nn.AdaptiveAvgPool3d((4, 80, 128)),
            nn.GroupNorm(32, 64),
            
            # Block 5
            nn.AdaptiveAvgPool3d((2, 64, 64)),
            nn.GroupNorm(32, 64),
            nn.Conv3d(64, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=1),
            nn.SELU()
        )
        
        # Latent space parameters
        self.mu_layer = nn.Conv3d(32, 32, kernel_size=1)
        self.logvar_layer = nn.Conv3d(32, 32, kernel_size=1)

    def _build_decoder(self) -> None:
        """Construct the decoder network."""
        self.decoder = nn.Sequential(
            # Initial layers
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=1),
            nn.GroupNorm(32, 64),
            nn.SELU(),
            
            # Upsample block 1
            nn.Upsample(size=(2, 64, 64), mode='nearest'),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=1),
            nn.GroupNorm(32, 64),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=1),
            
            # Upsample block 2
            nn.Upsample(size=(4, 80, 128), mode='nearest'),
            nn.ConvTranspose3d(64, 64, kernel_size=(1, 3, 1), 
                              stride=(1, 2, 1), padding=(0, 1, 0), 
                              output_padding=(0, 1, 0)),
            nn.GroupNorm(32, 64),
            nn.SELU(),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=1),
            nn.GroupNorm(32, 64),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=1),
            
            # Upsample block 3
            nn.ConvTranspose3d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=1),
            nn.GroupNorm(32, 64),
            nn.SELU(),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=1),
            nn.GroupNorm(32, 64),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=1),
            
            # Final upsampling
            nn.Upsample(size=(12, 360, 516), mode='nearest'),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=1),
            nn.GroupNorm(32, 64),
            nn.SELU(),
            nn.Conv3d(64, 1, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=1)
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input into latent space parameters.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Tuple of (mu, logvar) tensors
        """
        x = self.encoder(x)
        return self.mu_layer(x), self.logvar_layer(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector into reconstructed input.
        
        Args:
            z: Latent tensor
            
        Returns:
            Reconstructed tensor
        """
        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var).
        
        Args:
            mu: Mean of the latent Gaussian
            logvar: Log variance of the latent Gaussian
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the VAE.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (reconstructed input, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def forward_generative(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Generate multiple samples from the same input (generative mode).
        
        Args:
            x: Input tensor
            num_samples: Number of samples to generate
            
        Returns:
            Generated samples of shape (num_samples, B, C, D, H, W)
        """
        mu, logvar = self.encode(x)
        
        # Generate multiple samples
        samples = []
        for _ in range(num_samples):
            z = self.reparameterize(mu, logvar)
            sample = self.decode(z)
            samples.append(sample)
        
        return torch.stack(samples, dim=0)

    def generate_multiple_outputs(self, x: torch.Tensor, num_samples: int = 9) -> torch.Tensor:
        """
        Generate multiple diverse outputs for ensemble prediction.
        
        Args:
            x: Input tensor
            num_samples: Number of ensemble members to generate
            
        Returns:
            Ensemble predictions of shape (num_samples, B, C, D, H, W)
        """
        return self.forward_generative(x, num_samples)

    def freeze_encoder(self):
        """Freeze encoder parameters for generative fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.mu_layer.parameters():
            param.requires_grad = False
        for param in self.logvar_layer.parameters():
            param.requires_grad = False
        
        logger.info("Encoder parameters frozen for generative fine-tuning")

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.mu_layer.parameters():
            param.requires_grad = True
        for param in self.logvar_layer.parameters():
            param.requires_grad = True
        
        logger.info("Encoder parameters unfrozen")

    def get_trainable_parameters(self):
        """Get only trainable parameters (useful when encoder is frozen)."""
        return [param for param in self.parameters() if param.requires_grad]

# Alias for backward compatibility
DifferenceVAE = SimpleVAE3D_DifferenceGenerative