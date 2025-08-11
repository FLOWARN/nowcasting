# src/models/vae_finetuning.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging
from .vae import SimpleVAE3D

logger = logging.getLogger(__name__)

class SimpleVAE3D_GenerativeFinetuning(SimpleVAE3D):
    """3D Variational Autoencoder for converting deterministic to generative using CRPS fine-tuning."""
    
    def __init__(self, input_channels: int = 1, latent_dim: int = 32):
        """
        Initialize the 3D VAE model for fine-tuning.
        
        Args:
            input_channels: Number of input channels
            latent_dim: Dimension of latent space
        """
        super().__init__(input_channels, latent_dim)
        self.encoder_frozen = False
        
        logger.info(f"Initialized SimpleVAE3D_GenerativeFinetuning with input_channels={input_channels}, latent_dim={latent_dim}")

    def freeze_encoder(self):
        """Freeze encoder parameters for fine-tuning only the decoder."""
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Freeze latent space layers
        for param in self.mu_layer.parameters():
            param.requires_grad = False
        for param in self.logvar_layer.parameters():
            param.requires_grad = False
            
        self.encoder_frozen = True
        
        # Count frozen vs trainable parameters
        frozen_params = sum(p.numel() for p in self.encoder.parameters()) + \
                       sum(p.numel() for p in self.mu_layer.parameters()) + \
                       sum(p.numel() for p in self.logvar_layer.parameters())
        
        trainable_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        
        logger.info(f"Encoder frozen! Frozen parameters: {frozen_params:,}, Trainable parameters: {trainable_params:,}")
        
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters (for full training)."""
        # Unfreeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = True
            
        # Unfreeze latent space layers
        for param in self.mu_layer.parameters():
            param.requires_grad = True
        for param in self.logvar_layer.parameters():
            param.requires_grad = True
            
        self.encoder_frozen = False
        logger.info("Encoder unfrozen! All parameters are now trainable.")
        
    def load_pretrained_encoder(self, checkpoint_path: str, strict: bool = True):
        """
        Load pretrained encoder weights while keeping decoder trainable.
        
        Args:
            checkpoint_path: Path to pretrained model checkpoint
            strict: Whether to strictly enforce state dict matching
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract model state dict (handle DataParallel wrapper)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            # Handle DataParallel prefix
            if any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
            
            # Load the full model first
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"Missing keys when loading pretrained model: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys when loading pretrained model: {unexpected_keys}")
                
            logger.info(f"Loaded pretrained weights from: {checkpoint_path}")
            
            # Now freeze the encoder
            self.freeze_encoder()
            
        except Exception as e:
            logger.error(f"Error loading pretrained encoder: {str(e)}")
            raise
            
    def get_decoder_parameters(self):
        """Get only decoder parameters for optimization."""
        return self.decoder.parameters()
        
    def get_trainable_parameters(self):
        """Get all trainable parameters."""
        return filter(lambda p: p.requires_grad, self.parameters())
        
    def print_parameter_status(self):
        """Print status of model parameters."""
        print("\n=== Parameter Status ===")
        
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        encoder_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        
        mu_params = sum(p.numel() for p in self.mu_layer.parameters())
        mu_trainable = sum(p.numel() for p in self.mu_layer.parameters() if p.requires_grad)
        
        logvar_params = sum(p.numel() for p in self.logvar_layer.parameters())
        logvar_trainable = sum(p.numel() for p in self.logvar_layer.parameters() if p.requires_grad)
        
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        decoder_trainable = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        
        total_params = encoder_params + mu_params + logvar_params + decoder_params
        total_trainable = encoder_trainable + mu_trainable + logvar_trainable + decoder_trainable
        
        print(f"Encoder: {encoder_trainable:,}/{encoder_params:,} trainable")
        print(f"Mu Layer: {mu_trainable:,}/{mu_params:,} trainable")
        print(f"Logvar Layer: {logvar_trainable:,}/{logvar_params:,} trainable")
        print(f"Decoder: {decoder_trainable:,}/{decoder_params:,} trainable")
        print(f"Total: {total_trainable:,}/{total_params:,} trainable ({100*total_trainable/total_params:.1f}%)")
        print("========================\n")
        
    def forward_with_ensemble(self, x: torch.Tensor, num_samples: int = 20) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Memory-efficient forward pass that generates ensemble predictions.
        
        Args:
            x: Input tensor
            num_samples: Number of ensemble samples to generate
            
        Returns:
            Tuple of (reconstructed input, mu, logvar, ensemble predictions)
        """
        # Get latent parameters
        mu, logvar = self.encode(x)
        
        # Memory-efficient ensemble generation
        batch_size = x.shape[0]
        
        # Pre-allocate ensemble tensor
        ensemble = torch.empty((num_samples, *x.shape), device=x.device, dtype=x.dtype)
        
        # Generate ensemble predictions in small chunks to save memory
        chunk_size = min(3, num_samples)  # Process 3 samples at a time max
        
        for i in range(0, num_samples, chunk_size):
            end_idx = min(i + chunk_size, num_samples)
            current_chunk_size = end_idx - i
            
            # Generate predictions for this chunk
            for j in range(current_chunk_size):
                z = self.reparameterize(mu, logvar)
                pred = self.decode(z)
                ensemble[i + j] = pred
                del z, pred  # Explicit cleanup
            
            # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Generate single reconstruction for consistency
        z_single = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z_single)
        
        return reconstruction, mu, logvar, ensemble
        
    def forward_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using only the mean (deterministic output).
        
        Args:
            x: Input tensor
            
        Returns:
            Reconstructed tensor using mu only
        """
        mu, _ = self.encode(x)
        return self.decode(mu)
    
    def generate_multiple_outputs(self, x: torch.Tensor, num_samples: int = 10) -> torch.Tensor:
        """
        Generate multiple diverse outputs from the same input (the key generative capability).
        
        Args:
            x: Input tensor
            num_samples: Number of diverse outputs to generate
            
        Returns:
            Multiple outputs tensor of shape (num_samples, B, C, D, H, W)
        """
        # Get latent parameters from frozen encoder
        mu, logvar = self.encode(x)
        
        # Generate multiple diverse outputs
        outputs = []
        for _ in range(num_samples):
            # Sample from latent distribution
            z = self.reparameterize(mu, logvar)
            # Decode to get diverse output
            output = self.decode(z)
            outputs.append(output)
        
        return torch.stack(outputs, dim=0)
    
    def forward_generative(self, x: torch.Tensor, num_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for generative mode (samples from latent distribution).
        
        Args:
            x: Input tensor
            num_samples: Number of samples to generate
            
        Returns:
            Tuple of (sampled outputs, mu, logvar)
        """
        mu, logvar = self.encode(x)
        
        if num_samples == 1:
            # Single sample
            z = self.reparameterize(mu, logvar)
            output = self.decode(z)
            return output, mu, logvar
        else:
            # Multiple samples
            outputs = self.generate_multiple_outputs(x, num_samples)
            return outputs, mu, logvar