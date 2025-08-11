import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import TwoSlopeNorm
import os
from src.models.vae_finetuning import SimpleVAE3D_GenerativeFinetuning
from src.data_loading.vae_data_loader import VAEDataModule, SimpleSequenceDataset
from src.extrapolation_methods.langragian_persistance import langragian_persistance
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_trained_model(checkpoint_path, device):
    """Load the trained generative VAE model from checkpoint."""
    config = {
        'data': {'input_shape': [1, 12, 360, 516]},
        'model': {'latent_dim': 32}
    }
    
    model = SimpleVAE3D_GenerativeFinetuning(
        input_channels=config['data']['input_shape'][0],
        latent_dim=config['model']['latent_dim']
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # Handle DataParallel prefix
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded generative model from {checkpoint_path}")
    return model

def get_difference_sample(dataset, idx, window_size=12):
    """Get original difference for a specific sample."""
    # Ensure we don't go out of bounds
    if idx >= len(dataset) - 1:
        raise IndexError(f"Cannot get consecutive samples: idx {idx} too large for dataset size {len(dataset)}")
    
    # Get consecutive samples
    current_data = dataset[idx]
    next_data = dataset[idx + 1]
    
    # Convert to numpy if needed
    if hasattr(current_data, 'numpy'):
        current_data = current_data.numpy()
        next_data = next_data.numpy()
    
    # Debug: print shapes to understand the data format
    print(f"Current data shape: {current_data.shape}")
    print(f"Next data shape: {next_data.shape}")
    
    # Handle tensor format
    if len(current_data.shape) == 4:  # [time, channels, height, width]
        if current_data.shape[1] == 1:  # [time, 1, height, width]
            current_sample = current_data[:, 0, :, :]
            ground_truth_next = next_data[:, 0, :, :]
        else:  # [batch, time, height, width] or similar
            current_sample = current_data[0, :, :, :] if current_data.shape[0] == 1 else current_data
            ground_truth_next = next_data[0, :, :, :] if next_data.shape[0] == 1 else next_data
    elif len(current_data.shape) == 5:  # [batch, time, channels, height, width]
        current_sample = current_data[0, :, 0, :, :]
        ground_truth_next = next_data[0, :, 0, :, :]
    else:  # [time, height, width] or other format
        current_sample = current_data
        ground_truth_next = next_data
    
    # Compute Lagrangian persistence
    lagrangian_pred = langragian_persistance(current_sample, window_size)
    
    # Calculate difference
    difference = ground_truth_next - lagrangian_pred
    
    return difference, current_sample, ground_truth_next, lagrangian_pred

def create_comprehensive_gif(original_diff, reconstructed_diff, ground_truth, lagrangian_pred, sample_idx, output_dir, uncertainty, ensemble_members):
    """Create simple 4-panel visualization: GT, Lagrangian, GT-Langragian, VAE reconstructed."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the figure with 4 panels (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Sample {sample_idx}: Simple 4-Panel Comparison', fontsize=16, weight='bold')
    
    # Name the panels clearly
    ax1, ax2 = axes[0, 0], axes[0, 1]  # Top row: GT, Langragian
    ax3, ax4 = axes[1, 0], axes[1, 1]  # Bottom row: GT-Langragian, VAE reconstructed
    
    # Calculate color scales
    # For precipitation data (GT and Lagrangian)
    precip_max = max(ground_truth.max(), lagrangian_pred.max())
    precip_min = 0  # Precipitation can't be negative
    
    # For differences (original and reconstructed)
    diff_min = min(original_diff.min(), reconstructed_diff.min())
    diff_max = max(original_diff.max(), reconstructed_diff.max())
    diff_norm = TwoSlopeNorm(vmin=diff_min, vcenter=0, vmax=diff_max)
    
    # Initialize plots with clear names
    # Top row: Ground Truth | Lagrangian Prediction
    im1 = ax1.imshow(ground_truth[0], cmap='Blues', vmin=precip_min, vmax=precip_max, animated=True)
    ax1.set_title('Ground Truth\n(What Actually Happened)', fontsize=12, weight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    
    im2 = ax2.imshow(lagrangian_pred[0], cmap='Blues', vmin=precip_min, vmax=precip_max, animated=True)
    ax2.set_title('Lagrangian Prediction\n(Physics Baseline)', fontsize=12, weight='bold')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    
    # Bottom row: GT - Lagrangian | VAE Reconstructed
    im3 = ax3.imshow(original_diff[0], cmap='RdBu_r', norm=diff_norm, animated=True)
    ax3.set_title('GT - Langragian\n(True Residuals)', fontsize=12, weight='bold')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    
    im4 = ax4.imshow(reconstructed_diff[0], cmap='RdBu_r', norm=diff_norm, animated=True)
    ax4.set_title('VAE Reconstructed\n(Learned Residuals)', fontsize=12, weight='bold')
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')
    
    # Add colorbars
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Precipitation (mm/hr)', fontsize=10)
    
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Precipitation (mm/hr)', fontsize=10)
    
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label('Residuals (mm/hr)', fontsize=10)
    
    cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.8)
    cbar4.set_label('Residuals (mm/hr)', fontsize=10)
    
    # Add timestamp text
    time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=14, weight='bold')
    
    # Add simple statistics text
    def get_stats_text(frame):
        gt_mean = np.mean(ground_truth[frame])
        lag_mean = np.mean(lagrangian_pred[frame])
        diff_mean = np.mean(original_diff[frame])
        recon_mean = np.mean(reconstructed_diff[frame])
        
        # Calculate how well ML learned the residuals
        residual_error = np.mean(np.abs(original_diff[frame] - reconstructed_diff[frame]))
        residual_correlation = np.corrcoef(original_diff[frame].flatten(), reconstructed_diff[frame].flatten())[0, 1]
        
        stats = f"""Timestep {frame + 1}/12:
GT Mean: {gt_mean:.3f} mm/hr
Lagrangian Mean: {lag_mean:.3f} mm/hr
True Residuals Mean: {diff_mean:.3f} mm/hr
VAE Residuals Mean: {recon_mean:.3f} mm/hr
Residual Error (MAE): {residual_error:.3f} mm/hr
Residual Correlation: {residual_correlation:.3f}"""
        return stats
    
    stats_text = fig.text(0.02, 0.02, get_stats_text(0), fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.9))
    
    def animate(frame):
        im1.set_array(ground_truth[frame])
        im2.set_array(lagrangian_pred[frame])
        im3.set_array(original_diff[frame])
        im4.set_array(reconstructed_diff[frame])
        time_text.set_text(f'4-Panel Comparison - Timestep {frame + 1}/12')
        stats_text.set_text(get_stats_text(frame))
        return [im1, im2, im3, im4, time_text, stats_text]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=12, interval=800, blit=True, repeat=True
    )
    
    plt.tight_layout()
    
    # Save as GIF
    gif_path = os.path.join(output_dir, f'sample_{sample_idx:04d}_simple_4panel.gif')
    anim.save(gif_path, writer='pillow', fps=1.5)
    
    plt.close(fig)
    logger.info(f"Saved simple 4-panel GIF: {gif_path}")
    
    return gif_path

def create_summary_plots(original_diff, reconstructed_diff, sample_idx, output_dir, uncertainty, ensemble_members):
    """Create summary plots with statistics."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f'Sample {sample_idx}: Ensemble Analysis Summary', fontsize=16)
    
    # Calculate statistics
    mse = np.mean((original_diff - reconstructed_diff) ** 2)
    mae = np.mean(np.abs(original_diff - reconstructed_diff))
    correlation = np.corrcoef(original_diff.flatten(), reconstructed_diff.flatten())[0, 1]
    
    # Ensemble-specific statistics
    mean_uncertainty = np.mean(uncertainty)
    ensemble_diversity = np.mean(np.var(ensemble_members, axis=0))
    ensemble_spread = np.std(ensemble_members, axis=0)
    
    # Plot mean over time
    original_mean = np.mean(original_diff, axis=(1, 2))
    recon_mean = np.mean(reconstructed_diff, axis=(1, 2))
    uncertainty_mean = np.mean(uncertainty, axis=(1, 2))
    
    axes[0, 0].plot(original_mean, 'b-', label='Original', linewidth=2)
    axes[0, 0].plot(recon_mean, 'r--', label='Ensemble Mean', linewidth=2)
    axes[0, 0].fill_between(range(len(recon_mean)), 
                           recon_mean - uncertainty_mean, 
                           recon_mean + uncertainty_mean, 
                           alpha=0.3, color='red', label='Uncertainty')
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].set_ylabel('Mean Difference (mm/hr)')
    axes[0, 0].set_title('Temporal Evolution with Uncertainty')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot ensemble spread over time
    original_std = np.std(original_diff, axis=(1, 2))
    recon_std = np.std(reconstructed_diff, axis=(1, 2))
    
    axes[0, 1].plot(original_std, 'b-', label='Original', linewidth=2)
    axes[0, 1].plot(recon_std, 'r--', label='Ensemble Mean', linewidth=2)
    axes[0, 1].plot(uncertainty_mean, 'g:', label='Uncertainty', linewidth=2)
    axes[0, 1].set_xlabel('Timestep')
    axes[0, 1].set_ylabel('Standard Deviation (mm/hr)')
    axes[0, 1].set_title('Spatial Variability & Uncertainty')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Ensemble member spaghetti plot
    for i in range(min(5, len(ensemble_members))):  # Show first 5 members
        member_mean = np.mean(ensemble_members[i], axis=(1, 2))
        axes[0, 2].plot(member_mean, alpha=0.6, linewidth=1, color='gray')
    axes[0, 2].plot(recon_mean, 'r-', label='Ensemble Mean', linewidth=2)
    axes[0, 2].plot(original_mean, 'b-', label='Truth', linewidth=2)
    axes[0, 2].set_xlabel('Timestep')
    axes[0, 2].set_ylabel('Mean Difference (mm/hr)')
    axes[0, 2].set_title('Ensemble Member Diversity')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Time-averaged spatial patterns
    vmin = min(np.mean(original_diff, axis=0).min(), np.mean(reconstructed_diff, axis=0).min())
    vmax = max(np.mean(original_diff, axis=0).max(), np.mean(reconstructed_diff, axis=0).max())
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    im1 = axes[1, 0].imshow(np.mean(original_diff, axis=0), cmap='RdBu_r', norm=norm)
    axes[1, 0].set_title('Time-Averaged Original')
    axes[1, 0].set_xlabel('Longitude')
    axes[1, 0].set_ylabel('Latitude')
    plt.colorbar(im1, ax=axes[1, 0], shrink=0.8)
    
    im2 = axes[1, 1].imshow(np.mean(reconstructed_diff, axis=0), cmap='RdBu_r', norm=norm)
    axes[1, 1].set_title('Time-Averaged Ensemble Mean')
    axes[1, 1].set_xlabel('Longitude')
    axes[1, 1].set_ylabel('Latitude')
    plt.colorbar(im2, ax=axes[1, 1], shrink=0.8)
    
    # Uncertainty map
    im3 = axes[1, 2].imshow(np.mean(uncertainty, axis=0), cmap='Reds')
    axes[1, 2].set_title('Time-Averaged Uncertainty')
    axes[1, 2].set_xlabel('Longitude')
    axes[1, 2].set_ylabel('Latitude')
    plt.colorbar(im3, ax=axes[1, 2], shrink=0.8)
    
    # Error map
    error = np.mean(np.abs(original_diff - reconstructed_diff), axis=0)
    im4 = axes[2, 0].imshow(error, cmap='Reds')
    axes[2, 0].set_title('Time-Averaged Absolute Error')
    axes[2, 0].set_xlabel('Longitude')
    axes[2, 0].set_ylabel('Latitude')
    plt.colorbar(im4, ax=axes[2, 0], shrink=0.8)
    
    # Ensemble spread map
    im5 = axes[2, 1].imshow(np.mean(ensemble_spread, axis=0), cmap='Oranges')
    axes[2, 1].set_title('Time-Averaged Ensemble Spread')
    axes[2, 1].set_xlabel('Longitude')
    axes[2, 1].set_ylabel('Latitude')
    plt.colorbar(im5, ax=axes[2, 1], shrink=0.8)
    
    # Scatter plot of original vs reconstructed
    sample_indices = np.random.choice(original_diff.size, 10000, replace=False)
    orig_sample = original_diff.flatten()[sample_indices]
    recon_sample = reconstructed_diff.flatten()[sample_indices]
    
    axes[2, 2].scatter(orig_sample, recon_sample, alpha=0.5, s=1)
    min_val = min(orig_sample.min(), recon_sample.min())
    max_val = max(orig_sample.max(), recon_sample.max())
    axes[2, 2].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    axes[2, 2].set_xlabel('Original Difference (mm/hr)')
    axes[2, 2].set_ylabel('Ensemble Mean Difference (mm/hr)')
    axes[2, 2].set_title(f'Correlation (r = {correlation:.3f})')
    axes[2, 2].grid(True, alpha=0.3)
    
    # Add enhanced statistics text with ensemble metrics
    stats_text = f'''MSE: {mse:.6f}
MAE: {mae:.6f}
Correlation: {correlation:.3f}

Mean Uncertainty: {mean_uncertainty:.4f}
Ensemble Diversity: {ensemble_diversity:.4f}
Ensemble Spread: {np.mean(ensemble_spread):.4f}'''
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'sample_{sample_idx:04d}_summary.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved summary plot: {plot_path}")
    return plot_path, {'mse': mse, 'mae': mae, 'correlation': correlation, 
                       'mean_uncertainty': mean_uncertainty, 'ensemble_diversity': ensemble_diversity}

def main():
    # Configuration
    # For now, use the original deterministic model - update this path after fine-tuning completes
    checkpoint_path = "experiments_generative_crps/vae_generative_crps_20250714_135247/checkpoints/checkpoint_epoch_0.pth"
    output_dir = "visualization_outputs"
    h5_dataset_location = 'data/imerg_data.h5'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load trained model
    model = load_trained_model(checkpoint_path, device)
    
    # Load data
    data_provider = VAEDataModule(
        sequence_length=12,
        imerg_filename=h5_dataset_location,
        batch_size=1,
        image_shape=(360, 516),
        normalize_data=False,
        train_split=0.8,
        val_split=0.1
    )
    
    # Create datasets
    val_dataset = SimpleSequenceDataset(data_provider.val_sequences)
    
    # Visualize 5 random samples
    torch.manual_seed(42)  # For reproducible randomness
    max_idx = len(val_dataset) - 2  # Need room for consecutive samples
    samples_to_visualize = torch.randint(0, max_idx, (5,)).tolist()  # 5 random samples
    logger.info(f"Visualizing 5 random samples: {samples_to_visualize} (max valid idx: {max_idx})")
    all_stats = []
    
    for sample_idx in samples_to_visualize:
        logger.info(f"Processing sample {sample_idx}...")
        
        try:
            # Get original difference
            original_diff, current_sample, ground_truth, lagrangian_pred = get_difference_sample(
                val_dataset, sample_idx
            )
            
            # Get VAE reconstruction (ensemble=1)
            with torch.no_grad():
                # Convert to tensor and add batch dimension
                diff_tensor = torch.tensor(original_diff, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                
                # Generate single prediction (ensemble=1)
                ensemble_outputs = model.generate_multiple_outputs(diff_tensor, num_samples=1)
                
                # Get single output
                reconstructed_diff = ensemble_outputs[0].squeeze().cpu().numpy()
                uncertainty = np.zeros_like(reconstructed_diff)  # No uncertainty with single sample
                
                # Single member for consistency
                ensemble_members = ensemble_outputs.squeeze().cpu().numpy()[None, ...]  # Add batch dim
            
            # Create visualizations with ensemble data
            gif_path = create_comprehensive_gif(original_diff, reconstructed_diff, ground_truth, lagrangian_pred, sample_idx, output_dir, uncertainty, ensemble_members)
            plot_path, stats = create_summary_plots(original_diff, reconstructed_diff, sample_idx, output_dir, uncertainty, ensemble_members)
            
            all_stats.append(stats)
            
        except Exception as e:
            logger.error(f"Error processing sample {sample_idx}: {e}")
            continue
    
    # Print overall statistics
    if all_stats:
        avg_mse = np.mean([s['mse'] for s in all_stats])
        avg_mae = np.mean([s['mae'] for s in all_stats])
        avg_corr = np.mean([s['correlation'] for s in all_stats])
        avg_uncertainty = np.mean([s['mean_uncertainty'] for s in all_stats])
        avg_diversity = np.mean([s['ensemble_diversity'] for s in all_stats])
        
        logger.info(f"\nOverall Ensemble Statistics:")
        logger.info(f"Average MSE: {avg_mse:.6f}")
        logger.info(f"Average MAE: {avg_mae:.6f}")
        logger.info(f"Average Correlation: {avg_corr:.3f}")
        logger.info(f"Average Uncertainty: {avg_uncertainty:.4f}")
        logger.info(f"Average Ensemble Diversity: {avg_diversity:.4f}")
    
    logger.info(f"Visualization complete! Check {output_dir} for results.")

if __name__ == "__main__":
    main()