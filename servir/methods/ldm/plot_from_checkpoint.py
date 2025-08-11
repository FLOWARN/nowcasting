#!/usr/bin/env python3
"""
Standalone script to generate GT vs predicted_differences+langragian plots from a trained checkpoint.
This can be used to visualize results from any saved checkpoint without retraining.
"""

import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import h5py

# Add the ldm_data_loader directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ldm_data_loader'))
from langragian_ldm_dataloader import get_langragian_ldm_dataloader

# Add Stage1 for extrapolation methods
sys.path.append(os.path.join(os.path.dirname(__file__), 'residual_vae', 'src'))
from extrapolation_methods.langragian_persistance import langragian_persistance

from vae_models.vae_difference_generative import SimpleVAE3D_DifferenceGenerative
from generative.inferers.inferer import LatentDiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

def parse_args():
    parser = argparse.ArgumentParser(description='Generate plots from LDM checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the checkpoint file (e.g., checkpoints/ldm_epoch_4.pt)')
    parser.add_argument('--output-dir', type=str, default='checkpoint_plots',
                        help='Directory to save generated plots (default: checkpoint_plots)')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of random samples to plot (default: 10)')
    parser.add_argument('--inference-steps', type=int, default=1000,
                        help='Number of DDIM inference steps (default: 1000)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for data loading (default: 1)')
    return parser.parse_args()

def load_model_and_vae(checkpoint_path, device):
    """Load the trained LDM model and VAE from checkpoint."""
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Initialize U-Net with same architecture as training
    unet = DiffusionModelUNet(
        with_conditioning=True,
        cross_attention_dim=64,
        spatial_dims=3,
        in_channels=32,
        out_channels=32,
        num_res_blocks=6,
        num_channels=(128, 256, 512, 512),
        attention_levels=(False, True, True, True),
        num_head_channels=(0, 256, 512, 512),
    ).to(device)
    
    # Load model state
    unet.load_state_dict(checkpoint['model_state_dict'])
    unet.eval()
    
    # Initialize scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
        beta_start=0.0001,
        beta_end=0.02
    )
    
    # Load VAE model
    vae_checkpoint = os.path.join(os.path.dirname(__file__), 'vae_models', 'diff_encoder.pth')
    vae_model = SimpleVAE3D_DifferenceGenerative(input_channels=1, latent_dim=32).to(device)
    if os.path.exists(vae_checkpoint):
        vae_checkpoint_data = torch.load(vae_checkpoint, map_location=device, weights_only=False)
        if 'model_state_dict' in vae_checkpoint_data:
            vae_model.load_state_dict(vae_checkpoint_data['model_state_dict'])
        else:
            vae_model.load_state_dict(vae_checkpoint_data)
        print(f"Loaded VAE from {vae_checkpoint}")
    else:
        print(f"Warning: VAE checkpoint not found at {vae_checkpoint}")
    
    vae_model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint.get('loss', 'unknown'):.6f}")
    
    return unet, scheduler, vae_model

def get_dataloader(device, batch_size=1):
    """Get the data loader for testing."""
    
    imerg_file = os.path.join(os.path.dirname(__file__), 'ldm_data_loader', 'imerg_data.h5')
    ir_file = os.path.join(os.path.dirname(__file__), 'ldm_data_loader', 'filled_missing_nan_ir_data.h5')
    vae_checkpoint = os.path.join(os.path.dirname(__file__), 'vae_models', 'diff_encoder.pth')
    
    # Check if files exist
    for file_path, name in [(imerg_file, 'IMERG'), (ir_file, 'IR'), (vae_checkpoint, 'VAE')]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{name} file not found: {file_path}")
    
    # Create dataloader (dataloader handles both difference and IR VAE loading internally)
    full_loader = get_langragian_ldm_dataloader(
        imerg_file=imerg_file,
        ir_file=ir_file,
        vae_checkpoint_path=vae_checkpoint,
        batch_size=batch_size,
        shuffle=True,  # Shuffle for random samples
        num_workers=0,
        device=device,
        ir_start_offset=44,
        ir_steps=16,
        ir_stride=1,
        normalize_data=False  # Use raw data for plotting
    )
    
    return full_loader

def generate_prediction(model, scheduler, sample_context, sample_target, device, inference_steps=50):
    """Generate prediction using DDIM sampling."""
    
    with torch.no_grad():
        # Format context and target
        sample_context_formatted = sample_context.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        sample_target_formatted = sample_target.permute(0, 2, 1, 3, 4)    # (B, C, T, H, W)
        
        # Generate prediction using DDIM sampling
        latent_sample = torch.randn_like(sample_target_formatted).to(device)
        
        # DDIM sampling
        step_size = scheduler.num_train_timesteps // inference_steps
        
        for i, t in enumerate(range(scheduler.num_train_timesteps - 1, -1, -step_size)):
            t_tensor = torch.tensor([t], device=device).long()
            
            # Predict noise
            noise_pred = model(
                x=latent_sample,
                timesteps=t_tensor,
                context=sample_context_formatted
            )
            
            # DDIM step
            alpha_t = scheduler.alphas_cumprod[t]
            alpha_prev = scheduler.alphas_cumprod[max(0, t - step_size)]
            
            pred_x0 = (latent_sample - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            latent_sample = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * noise_pred
    
    return latent_sample

def get_original_data_from_sample(sample):
    """Extract original data directly from dataloader sample."""
    gt_precipitation = sample['gt_target'].detach().cpu().numpy()  # GT[25:36]
    langragian_prediction = sample['langragian_target'].detach().cpu().numpy()  # LP[25:36]
    true_differences = sample['diff_target'].detach().cpu().numpy()  # GT[25:36] - LP[25:36] (true differences)
    return gt_precipitation, langragian_prediction, true_differences

def create_visualization(gt_precipitation, corrected_prediction, langragian_prediction, 
                        predicted_differences, true_differences, output_path, sample_idx, epoch):
    """Create 5-panel visualization and save as PNG and GIF."""
    
    n_timesteps = min(12, gt_precipitation.shape[0])
    
    # Create individual frames for GIF
    frame_paths = []
    for t in range(n_timesteps):
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        
        # Ground Truth Precipitation
        im1 = axes[0].imshow(gt_precipitation[t], cmap='YlGnBu', vmin=0, vmax=20)
        axes[0].set_title(f'Ground Truth (t={t})', fontsize=12)
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='mm/hr')
        
        # Corrected Prediction (Langragian + Predicted Differences)
        im2 = axes[1].imshow(corrected_prediction[t], cmap='YlGnBu', vmin=0, vmax=20)
        axes[1].set_title(f'Corrected Prediction (t={t})', fontsize=12)
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='mm/hr')
        
        # Raw Langragian Prediction
        im3 = axes[2].imshow(langragian_prediction[t], cmap='YlGnBu', vmin=0, vmax=20)
        axes[2].set_title(f'Raw Langragian (t={t})', fontsize=12)
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04, label='mm/hr')
        
        # Predicted Differences (AI model output)
        diff_vmin, diff_vmax = -10, 10  # Differences can be negative
        im4 = axes[3].imshow(predicted_differences[t], cmap='RdBu_r', vmin=diff_vmin, vmax=diff_vmax)
        axes[3].set_title(f'Predicted Diff (t={t})', fontsize=12)
        axes[3].axis('off')
        plt.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04, label='mm/hr')
        
        # True Differences (Ground Truth - Langragian)
        # Handle different tensor shapes - true_differences might be (T, H, W) or (T, C, H, W)
        if len(true_differences.shape) == 4:  # (T, C, H, W)
            true_diff_slice = true_differences[t, 0]
        else:  # (T, H, W)
            true_diff_slice = true_differences[t]
        im5 = axes[4].imshow(true_diff_slice, cmap='RdBu_r', vmin=diff_vmin, vmax=diff_vmax)
        axes[4].set_title(f'True Diff (t={t})', fontsize=12)
        axes[4].axis('off')
        plt.colorbar(im5, ax=axes[4], fraction=0.046, pad=0.04, label='mm/hr')
        
        # Add sample and epoch info
        fig.suptitle(f'Sample {sample_idx:02d} - Timestep {t:02d}/12 (Epoch {epoch})', 
                    fontsize=16, y=0.95)
        
        plt.tight_layout()
        
        # Save frame
        frame_path = output_path.replace('.png', f'_timestep_{t:02d}.png')
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()
        frame_paths.append(frame_path)
    
    # Create GIF from frames
    try:
        images = [Image.open(frame_path) for frame_path in frame_paths]
        gif_path = output_path.replace('.png', '.gif')
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=500,  # 500ms per frame
            loop=0
        )
        print(f"  -> Created GIF: {os.path.basename(gif_path)}")
        
        # Clean up individual frame files
        for frame_path in frame_paths:
            os.remove(frame_path)
            
    except ImportError:
        print("  -> PIL not available, skipping GIF creation")

def main():
    args = parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and VAE
    model, scheduler, vae_model = load_model_and_vae(args.checkpoint, device)
    
    # Get dataloader
    dataloader = get_dataloader(device, args.batch_size)
    
    # Extract epoch from checkpoint for naming
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    epoch = checkpoint.get('epoch', 'unknown')
    
    print(f"Generating {args.num_samples} sample visualizations...")
    
    # Generate visualizations for random samples
    for sample_num in range(args.num_samples):
        print(f"Processing sample {sample_num + 1}/{args.num_samples}")
        
        # Clear GPU cache before each sample
        torch.cuda.empty_cache()
        
        # Get random sample
        sample_idx = torch.randint(0, len(dataloader.dataset), (1,)).item()
        
        with torch.no_grad():  # Disable gradients for inference
            sample = dataloader.dataset[sample_idx]
        
            # Move to device and add batch dimension
            sample_context = sample['context'].unsqueeze(0).to(device)
            sample_target = sample['target'].unsqueeze(0).to(device)
        
            # Generate prediction
            predicted_latent = generate_prediction(
                model, scheduler, sample_context, sample_target, device, args.inference_steps
            )
        
            # Decode prediction
            try:
                # Format for VAE decoding
                pred_diff_latent = predicted_latent  # Already in (B, C, T, H, W)
                
                # Decode differences
                pred_diff_decoded = vae_model.decode(pred_diff_latent)
                
                # Get original data directly from sample
                gt_precipitation, langragian_prediction, true_differences = get_original_data_from_sample(sample)
                
                # Convert predictions to numpy
                predicted_differences = pred_diff_decoded.detach().cpu().numpy()  # Keep batch dim for now
                print(f"    VAE output shape: {predicted_differences.shape}")
                
                # VAE outputs (B, C, T, H, W), we need (T, H, W)
                if len(predicted_differences.shape) == 5:  # (B, C, T, H, W)
                    predicted_differences_fixed = predicted_differences[0, 0]  # Remove batch and channel: (T, H, W)
                elif len(predicted_differences.shape) == 4:  # (B, T, H, W) or (T, C, H, W)
                    if predicted_differences.shape[1] == 1:  # (T, C, H, W)
                        predicted_differences_fixed = predicted_differences[:, 0]  # (T, H, W)
                    else:  # (B, T, H, W)
                        predicted_differences_fixed = predicted_differences[0]  # (T, H, W)
                else:
                    predicted_differences_fixed = predicted_differences
                
                # Final corrected prediction = Langragian + Predicted_differences
                corrected_prediction = langragian_prediction + predicted_differences_fixed
                
                # Create visualization
                output_path = os.path.join(args.output_dir, f'sample_{sample_num+1:02d}_epoch_{epoch}.png')
                create_visualization(gt_precipitation, corrected_prediction, langragian_prediction,
                                   predicted_differences_fixed, true_differences, output_path, sample_num+1, epoch)
                
                print(f"  -> Saved: {os.path.basename(output_path)}")
                
            except Exception as e:
                print(f"  -> Failed to process sample {sample_num + 1}: {e}")
                continue
        
        # Clear GPU cache after each sample
        torch.cuda.empty_cache()
    
    print(f"\nCompleted! Check {args.output_dir} for generated visualizations.")
    print("Files generated:")
    print("  - sample_XX_epoch_Y.gif (animated sequences)")
    print("\nVisualization panels:")
    print("  1. Ground Truth - actual precipitation")
    print("  2. Corrected Prediction - Langragian + AI corrections")
    print("  3. Raw Langragian - baseline method")
    print("  4. Predicted Differences - AI model output")
    print("  5. True Differences - what the AI should have predicted")

if __name__ == "__main__":
    main()