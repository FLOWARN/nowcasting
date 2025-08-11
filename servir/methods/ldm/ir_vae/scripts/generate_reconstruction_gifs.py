#!/usr/bin/env python3
"""
Script to generate GIF plots comparing original and reconstructed IR images from VAE model.
Creates separate GIFs for each sample with frame numbers for better understanding.
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import h5py
from PIL import Image
import logging

# Import model and data utilities
from src.models.vae import SimpleVAE3D
from src.data.data_provider import IMERGDataModule
from src.utils.config import load_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_custom_colormap():
    """Create a custom colormap for IR temperature visualization"""
    colors = ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000', '#800000']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('ir_temp', colors, N=n_bins)
    return cmap

def load_model(checkpoint_path, config):
    """Load the trained VAE model from checkpoint."""
    model = SimpleVAE3D(
        input_channels=config['data']['input_shape'][0],
        latent_dim=config['model']['latent_dim']
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Extract mean and std from checkpoint
    mean = checkpoint.get('mean', 0.0)
    std = checkpoint.get('std', 1.0)
    
    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"Normalization parameters - Mean: {mean}, Std: {std}")
    
    return model, mean, std

def prepare_data(config):
    """Prepare data loader for generating reconstructions."""
    ir_h5_dataset_location = '../ldm_data_loader/filled_missing_nan_ir_data.h5'
    
    data_provider = IMERGDataModule(
        forecast_steps=12,
        history_steps=12,
        ir_filename=ir_h5_dataset_location,
        batch_size=1,  # Process one sample at a time
        image_shape=(360, 516),
        normalize_data=False,
        dataset='wa_ir',
        production_mode=False,
    )
    
    return data_provider.test_dataloader()

def create_comparison_frame(original, reconstructed, frame_idx, sample_idx, vmin=0, vmax=1):
    """Create a single frame comparing original vs reconstructed image."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Create custom colormap
    cmap = create_custom_colormap()
    
    # Original image
    im1 = ax1.imshow(original, cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set_title(f'Original - Frame {frame_idx}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    plt.colorbar(im1, ax=ax1, label='Temperature (normalized)')
    
    # Reconstructed image
    im2 = ax2.imshow(reconstructed, cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.set_title(f'Reconstructed - Frame {frame_idx}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    plt.colorbar(im2, ax=ax2, label='Temperature (normalized)')
    
    # Add sample info
    fig.suptitle(f'Sample {sample_idx:04d} - Original vs Reconstructed IR', 
                 fontsize=16, fontweight='bold')
    
    # Add frame number annotation
    fig.text(0.5, 0.02, f'Frame: {frame_idx}/16', ha='center', 
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    return fig

def generate_gif_for_sample(model, input_data, sample_idx, output_dir, device, mean, std):
    """Generate GIF for a single sample showing original vs reconstructed sequences."""
    model.to(device)
    
    with torch.no_grad():
        # Prepare input data
        original = input_data.clone()
        input_tensor = original.to(device).float()
        
        # Apply permutation and normalization like in visualization.py
        input_tensor = input_tensor.permute(0, 1, 4, 2, 3)  # -> (B, C, H, T, W)
        
        # Move mean/std to device if they're tensors
        if isinstance(mean, torch.Tensor):
            mean = mean.to(device)
        if isinstance(std, torch.Tensor):
            std = std.to(device)
        
        input_tensor_norm = (input_tensor - mean) / std
        
        # Get reconstruction
        reconstructed, _, _ = model(input_tensor_norm)
        
        # Move back to CPU and convert to numpy
        original_np = input_tensor.squeeze().cpu().numpy()  # Shape: [16, 360, 516]
        reconstructed_np = reconstructed.squeeze().cpu().detach().numpy()  # Shape: [16, 360, 516]
        
        # Check if shapes need permutation
        if original_np.shape[0] != 16:
            # Permute to get time dimension first
            if len(original_np.shape) == 4:  # [C, T, H, W]
                original_np = original_np.transpose(1, 0, 2, 3).squeeze()
            elif len(original_np.shape) == 3 and original_np.shape[-1] == 16:  # [H, W, T]
                original_np = original_np.transpose(2, 0, 1)
            elif len(original_np.shape) == 3 and original_np.shape[0] == 16:  # Already [T, H, W]
                pass
            else:
                print(f"Warning: Unexpected original shape {original_np.shape}, trying transpose...")
                original_np = original_np.transpose()
        
        if reconstructed_np.shape[0] != 16:
            if len(reconstructed_np.shape) == 4:  # [C, T, H, W]
                reconstructed_np = reconstructed_np.transpose(1, 0, 2, 3).squeeze()
            elif len(reconstructed_np.shape) == 3 and reconstructed_np.shape[-1] == 16:  # [H, W, T]
                reconstructed_np = reconstructed_np.transpose(2, 0, 1)
            elif len(reconstructed_np.shape) == 3 and reconstructed_np.shape[0] == 16:  # Already [T, H, W]
                pass
            else:
                print(f"Warning: Unexpected reconstructed shape {reconstructed_np.shape}, trying transpose...")
                reconstructed_np = reconstructed_np.transpose()
        
        print(f"Final shapes - Original: {original_np.shape}, Reconstructed: {reconstructed_np.shape}")
        
        # Denormalize reconstruction using saved mean/std
        if isinstance(mean, torch.Tensor):
            mean_cpu = mean.cpu().numpy()
        else:
            mean_cpu = mean
        if isinstance(std, torch.Tensor):
            std_cpu = std.cpu().numpy()
        else:
            std_cpu = std
            
        reconstructed_np = reconstructed_np * std_cpu + mean_cpu
        
        # For visualization, normalize both to 0-1 range for consistent display
        original_display = (original_np - original_np.min()) / (original_np.max() - original_np.min() + 1e-8)
        reconstructed_display = (reconstructed_np - reconstructed_np.min()) / (reconstructed_np.max() - reconstructed_np.min() + 1e-8)
        
        # Create frames
        frames = []
        temp_dir = os.path.join(output_dir, f'temp_frames_sample_{sample_idx:04d}')
        os.makedirs(temp_dir, exist_ok=True)
        
        logger.info(f"Creating frames for sample {sample_idx}...")
        
        for frame_idx in range(16):
            fig = create_comparison_frame(
                original_display[frame_idx], 
                reconstructed_display[frame_idx], 
                frame_idx + 1, 
                sample_idx,
                vmin=0, 
                vmax=1
            )
            
            # Save frame
            frame_path = os.path.join(temp_dir, f'frame_{frame_idx:02d}.png')
            fig.savefig(frame_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            frames.append(Image.open(frame_path))
        
        # Create GIF
        gif_path = os.path.join(output_dir, f'sample_{sample_idx:04d}_reconstruction.gif')
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=800,  # 800ms per frame
            loop=0
        )
        
        # Clean up temporary frames
        for frame_file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, frame_file))
        os.rmdir(temp_dir)
        
        logger.info(f"Saved GIF: {gif_path}")
        return gif_path

def main():
    parser = argparse.ArgumentParser(description='Generate reconstruction GIFs from VAE model')
    parser.add_argument('--checkpoint', type=str, 
                       default='outputs/vae_3d_experiment_20250616_163638/checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, 
                       default='configs/train_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to generate GIFs for')
    parser.add_argument('--output_dir', type=str, default='reconstruction_gifs',
                       help='Output directory for GIFs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for inference')
    parser.add_argument('--dataset_split', type=str, default='test', choices=['train', 'val', 'test'],
                       help='Which dataset split to use for visualization')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load model
        logger.info("Loading model...")
        model, mean, std = load_model(args.checkpoint, config)
        
        # Prepare data
        logger.info("Preparing data...")
        dataloader = prepare_data(config)
        
        # Generate GIFs
        logger.info(f"Generating GIFs for {args.num_samples} samples...")
        
        gif_paths = []
        for i, batch in enumerate(dataloader):
            if i >= args.num_samples:
                break
                
            # Extract input data (ignore output since we're doing reconstruction)
            input_data, _ = batch
            
            logger.info(f"Processing sample {i+1}/{args.num_samples}")
            
            gif_path = generate_gif_for_sample(
                model, 
                input_data, 
                i, 
                args.output_dir, 
                args.device,
                mean,
                std
            )
            gif_paths.append(gif_path)
        
        # Create summary file
        summary_path = os.path.join(args.output_dir, 'generation_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Generated {len(gif_paths)} reconstruction GIFs\n")
            f.write(f"Model: {args.checkpoint}\n")
            f.write(f"Configuration: {args.config}\n")
            f.write(f"Device: {args.device}\n")
            f.write(f"Mean: {mean}, Std: {std}\n\n")
            f.write("Generated files:\n")
            for gif_path in gif_paths:
                f.write(f"  - {os.path.basename(gif_path)}\n")
        
        logger.info(f"All GIFs generated successfully!")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Generated {len(gif_paths)} GIFs")
        
    except Exception as e:
        logger.error(f"Error generating GIFs: {str(e)}")
        raise

if __name__ == "__main__":
    main()