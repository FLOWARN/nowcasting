import torch
import torch.nn.functional as F

def compute_sample_weighted_loss(noise_pred, noise_target, original_precip=None, weight_factor=4.0, threshold=0.1):
    """
    MSE loss with higher weights for samples (entire images) with more precipitation.
    
    Args:
        noise_pred: Predicted noise (B, C, T, H, W)
        noise_target: Target noise (B, C, T, H, W) 
        original_precip: Original precipitation data (B, 1, T, H, W) in mm/hr
        weight_factor: How much more weight to give high-precipitation samples
        threshold: Precipitation threshold in mm/hr
    """
    # Base MSE loss per sample
    base_loss = F.mse_loss(noise_pred, noise_target, reduction='none')  # (B, C, T, H, W)
    sample_losses = base_loss.view(base_loss.size(0), -1).mean(dim=1)  # (B,)
    
    if original_precip is not None:
        # Calculate average precipitation per sample
        sample_precip = original_precip.view(original_precip.size(0), -1).mean(dim=1)  # (B,)
        
        # Create weight based on precipitation amount per sample
        # Samples with more precipitation get higher weight
        precip_weight = (sample_precip > threshold).float()
        sample_weights = 1.0 + (weight_factor - 1.0) * precip_weight  # (B,)
        
        # Apply weights to sample losses
        weighted_losses = sample_losses * sample_weights
        return weighted_losses.mean()
    else:
        # If no original data, fall back to regular MSE
        return sample_losses.mean()

def compute_loss(noise_pred, noise_target, loss_type='mse', original_data=None, precip_weight=4.0, precip_threshold=0.1):
    """
    Compute loss for precipitation nowcasting.
    
    Args:
        noise_pred: Predicted noise (B, C, T, H, W)
        noise_target: Target noise (B, C, T, H, W)
        loss_type: 'mse', 'l1', 'l1+l2', or 'weighted_mse'
        original_data: Original precipitation data for weighting
        precip_weight: Weight factor for high-precipitation samples
        precip_threshold: Precipitation threshold in mm/hr
    """
    
    if loss_type == 'mse':
        return F.mse_loss(noise_pred, noise_target)
    
    elif loss_type == 'l1':
        return F.l1_loss(noise_pred, noise_target)
    
    elif loss_type == 'l1+l2':
        return 0.5 * F.mse_loss(noise_pred, noise_target) + 0.5 * F.l1_loss(noise_pred, noise_target)
    
    elif loss_type == 'weighted_mse':
        return compute_sample_weighted_loss(
            noise_pred, noise_target, original_data, 
            weight_factor=precip_weight, threshold=precip_threshold
        )
    
    else:
        # Fallback to MSE
        return F.mse_loss(noise_pred, noise_target)