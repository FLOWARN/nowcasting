#!/usr/bin/env python3
"""
CRPS/CRPSS Calculation Script for Latent Difference Precipitation Nowcasting

This script calculates Continuous Ranked Probability Score (CRPS) and CRPS Skill Score (CRPSS) 
for LDM trained on difference between IMERG ground truth and Lagrangian prediction error in latent space.

The LDM is trained to predict: (IMERG_GT - Lagrangian_Prediction_Error) in latent space
Final prediction = Lagrangian_Baseline + LDM_Prediction

Uses CRPS fine-tuned generative VAE decoder for non-deterministic reconstruction
with ensemble_size=9 for all methods for proper uncertainty quantification.

Compares LDM predictions with baseline methods (STEPS, LINDA, Lagrangian, Naive).
"""

import os
import sys
import torch
import argparse
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from tqdm import tqdm
import h5py
import pickle

# CRPS calculation imports
try:
    import properscoring as ps
    CRPS_AVAILABLE = True
    print("✓ properscoring imported successfully")
except ImportError:
    print("Warning: properscoring not available. Installing...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "properscoring"])
        import properscoring as ps
        CRPS_AVAILABLE = True
        print("✓ properscoring installed and imported successfully")
    except:
        print("Failed to install properscoring. CRPS calculations will be unavailable.")
        CRPS_AVAILABLE = False

# PySTEPS imports
try:
    from pysteps.utils import transformation
    from pysteps import nowcasts
    from pysteps import motion
    from pysteps.motion.lucaskanade import dense_lucaskanade
    PYSTEPS_AVAILABLE = True
    print("✓ PySTEPS imported successfully")
except ImportError as e:
    print(f"Warning: PySTEPS not available. PySTEPS baseline will be skipped. Error: {e}")
    PYSTEPS_AVAILABLE = False

# Add the ldm_data_loader directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ldm_data_loader'))
from langragian_ldm_dataloader import get_langragian_ldm_dataloader

# Import LDM components
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

# Import CRPS fine-tuned generative VAE
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'residual_vae', 'src'))
from models.vae_finetuning import SimpleVAE3D_GenerativeFinetuning

# Since we're using PySTEPS directly for STEPS and LINDA, we don't need custom imports
# PySTEPS already imported above, so we can use it directly
NOWCASTING_AVAILABLE = PYSTEPS_AVAILABLE  # Use STEPS availability
if NOWCASTING_AVAILABLE:
    print("✓ Nowcasting methods available via PySTEPS")
else:
    print("Warning: Nowcasting methods not available (PySTEPS missing)")

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate CRPS/CRPSS for Latent Difference LDM Nowcasting')
    
    # Dataset parameters
    parser.add_argument('--num-samples', type=int, default=150,
                        help='Number of random test samples to use (default: 100)')
    parser.add_argument('--seed', type=int, default=11,
                        help='Random seed for reproducible sample selection (default: 11)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for inference (default: 1)')
    
    # Model parameters
    parser.add_argument('--ldm-checkpoint', type=str, 
                        default='checkpoints/ldm_epoch_4.pt',
                        help='Path to pretrained LDM checkpoint')
    parser.add_argument('--vae-checkpoint', type=str,
                        default='residual_vae/experiments_generative_crps/vae_generative_crps_20250714_135247/checkpoints/best_model.pth',
                        help='Path to CRPS fine-tuned generative VAE checkpoint')
    
    # Inference parameters
    parser.add_argument('--num-inference-steps', type=int, default=100,
                        help='Number of DDPM sampling steps (default: 100)')
    parser.add_argument('--ensemble-size', type=int, default=9,
                        help='Number of ensemble predictions to generate (default: 9)')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='crps_crpss_results_latent_diff',
                        help='Directory to save CRPS results and plots')
    
    # Checkpointing options
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save all predictions to HDF5 file for later use')
    parser.add_argument('--predictions-file', type=str, default='crps_predictions_latent_diff.h5',
                        help='HDF5 file to save/load predictions')
    parser.add_argument('--load-predictions', action='store_true',
                        help='Load predictions from file and skip inference')
    parser.add_argument('--checkpoint-interval', type=int, default=1,
                        help='Save checkpoint every N samples (default: 1)')

    return parser.parse_args()

def load_ldm_model(checkpoint_path, device):
    """Load LDM model"""
    print(f"Loading LDM model from: {checkpoint_path}")
    
    # Initialize U-Net - matching training configuration
    unet = DiffusionModelUNet(
        with_conditioning=True,
        cross_attention_dim=64,  # Context has both difference + IR encoded features
        spatial_dims=3,
        in_channels=32,          # Target latent channels from difference VAE
        out_channels=32,         # Target latent channels from difference VAE
        num_res_blocks=6,
        num_channels=(128, 256, 512, 512),
        attention_levels=(False, True, True, True),
        num_head_channels=(0, 256, 512, 512),
    ).to(device)
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        unet.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        raise FileNotFoundError(f"LDM checkpoint not found: {checkpoint_path}")
    
    unet.eval()
    
    # Initialize scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
        beta_start=0.0001,
        beta_end=0.02
    )
    
    return unet, scheduler

def load_crps_finetuned_vae(checkpoint_path, device):
    """Load CRPS fine-tuned generative VAE"""
    print(f"Loading CRPS fine-tuned generative VAE from: {checkpoint_path}")
    
    # Initialize the generative VAE model
    vae_model = SimpleVAE3D_GenerativeFinetuning(
        input_channels=1,
        latent_dim=32
    ).to(device)
    
    # Load the fine-tuned checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Handle DataParallel prefix if present
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
        
        # Load weights
        missing_keys, unexpected_keys = vae_model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"  Warning: Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"  Warning: Unexpected keys: {unexpected_keys}")
        
        print(f"  ✓ CRPS fine-tuned VAE loaded successfully")
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"  ✓ Checkpoint from epoch: {epoch}")
        
    else:
        raise FileNotFoundError(f"CRPS VAE checkpoint not found: {checkpoint_path}")
    
    vae_model.eval()
    return vae_model

def get_test_data(vae_checkpoint_path, device, num_samples, seed):
    """Get test data using LangragianLDMDataset"""
    print(f"Setting up test data with {num_samples} random samples (seed={seed})")
    
    imerg_file = './ldm_data_loader/imerg_data.h5'
    ir_file = './ldm_data_loader/filled_missing_nan_ir_data.h5'
    
    # Create dataloader using LangragianLDMDataset
    full_loader = get_langragian_ldm_dataloader(
        imerg_file=imerg_file,
        ir_file=ir_file,
        vae_checkpoint_path=vae_checkpoint_path,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        device=device,
        ir_start_offset=44,
        ir_steps=16,
        ir_stride=1
    )
    
    # Get test split
    dataset = full_loader.dataset
    total_len = len(dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len
    
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])
    
    print(f"Total available samples: {total_len}")
    print(f"Dataset splits - Train: {train_len}, Val: {val_len}, Test: {test_len}")
    
    # Get random test samples
    random.seed(seed)
    test_indices = random.sample(range(len(test_ds)), min(num_samples, len(test_ds)))
    print(f"Randomly selected {len(test_indices)} samples from test set")
    
    return test_ds, test_indices

def generate_ldm_ensemble_predictions(unet, scheduler, vae_model, dataset, sample_data, device, 
                                     num_inference_steps, ensemble_size=9):
    """
    Generate ensemble LDM predictions using the latent difference training paradigm
    with CRPS fine-tuned generative VAE for non-deterministic decoding.
    """
    ensemble_predictions = []
    
    with torch.no_grad():
        # Extract data from the sample (already processed by LangragianLDMDataset)
        context = sample_data['context']  # Combined encoded: diff + IR
        lagrangian_target = sample_data['langragian_target']  # LP[25:36] - our baseline
        
        print(f"    Generating {ensemble_size} LDM ensemble members...")
        
        # Convert Lagrangian target to tensor if needed
        if isinstance(lagrangian_target, np.ndarray):
            lagrangian_target_tensor = torch.from_numpy(lagrangian_target).float()
        else:
            lagrangian_target_tensor = lagrangian_target
        
        # Add channel dimension if needed
        if len(lagrangian_target_tensor.shape) == 3:  # (T, H, W)
            lagrangian_target_tensor = lagrangian_target_tensor.unsqueeze(1)  # (T, 1, H, W)
        
        for ensemble_idx in range(ensemble_size):
            if ensemble_size > 1:
                print(f"      Generating ensemble member {ensemble_idx + 1}/{ensemble_size}")
            
            # Permute context to match training format: (B, T, C, H, W) -> (B, C, T, H, W)
            context_formatted = context.permute(0, 2, 1, 3, 4).to(device)
            
            # Initialize random noise for the difference prediction
            target_shape = sample_data['target'].shape  # Encoded target shape
            latent_sample = torch.randn(target_shape).to(device)
            latent_sample = latent_sample.permute(0, 2, 1, 3, 4)  # Match format: (B, C, T, H, W)
            
            # DDPM sampling for the difference
            step_interval = max(1, scheduler.num_train_timesteps // num_inference_steps)
            timesteps = list(range(scheduler.num_train_timesteps - 1, -1, -step_interval))[:num_inference_steps]
            
            for t in timesteps:
                t_tensor = torch.tensor([t], device=device).long()
                
                # Predict noise for the difference
                noise_pred = unet(
                    x=latent_sample,
                    timesteps=t_tensor,
                    context=context_formatted
                )
                
                # Remove noise
                step_output = scheduler.step(noise_pred, t, latent_sample)
                if hasattr(step_output, 'prev_sample'):
                    latent_sample = step_output.prev_sample
                else:
                    latent_sample = step_output[0] if isinstance(step_output, tuple) else step_output
            
            # Decode the predicted difference using CRPS fine-tuned generative VAE
            predicted_diff_latent = latent_sample.permute(0, 2, 1, 3, 4)  # Back to (B, T, C, H, W)
            latent_for_decode = predicted_diff_latent.squeeze(0).permute(1, 0, 2, 3).unsqueeze(0).to(device)
            
            # Decode from latent space to precipitation space
            predicted_diff_decoded = vae_model.decode(latent_for_decode)
            predicted_diff_decoded = predicted_diff_decoded.squeeze(0).permute(1, 0, 2, 3)  # (T, C, H, W)
            
            # Remove predicted_diff batch dimension and ensure same shape
            if len(predicted_diff_decoded.shape) == 4:  # (T, C, H, W)
                predicted_diff_final = predicted_diff_decoded
            else:
                predicted_diff_final = predicted_diff_decoded.squeeze(0)
            
            # Final prediction = Lagrangian baseline + predicted difference
            final_prediction = lagrangian_target_tensor + predicted_diff_final.cpu()
            
            # Add batch dimension for consistency: (T, C, H, W) -> (1, T, C, H, W)
            final_prediction = final_prediction.unsqueeze(0)
            
            ensemble_predictions.append(final_prediction)
    
    return ensemble_predictions

def run_baseline_ensemble_methods(sample_data, ensemble_size=9, timesteps=12):
    """
    Run baseline nowcasting methods with ensemble generation.
    Each method uses its optimal input strategy and generates ensemble_size predictions.
    """
    baseline_predictions = {}
    
    if not NOWCASTING_AVAILABLE:
        print("Warning: Nowcasting methods not available")
        return baseline_predictions
    
    # Extract ground truth context (GT[13:24]) for baseline method input
    gt_context = sample_data['gt_context']  # GT[13:24] - available context
    
    # Convert to numpy if it's a tensor
    if torch.is_tensor(gt_context):
        gt_context_np = gt_context.cpu().numpy()
    else:
        gt_context_np = gt_context
    
    # Ensure shape is (T, H, W) - should already be correct from dataset
    print(f"    GT context shape before processing: {gt_context_np.shape}")
    if len(gt_context_np.shape) == 5:  # (B, T, C, H, W)
        gt_context_np = gt_context_np[0, :, 0, :, :]  # Take first batch and channel
    elif len(gt_context_np.shape) == 4:  # (B, T, H, W) or (T, C, H, W)
        if gt_context_np.shape[0] == 1:  # (B, T, H, W) where B=1
            gt_context_np = gt_context_np[0, :, :, :]  # Remove batch dimension -> (T, H, W)
        else:  # (T, C, H, W)
            gt_context_np = gt_context_np[:, 0, :, :]  # Remove channel dimension
    elif len(gt_context_np.shape) == 3:  # Already (T, H, W)
        pass  # No change needed
    
    print(f"    Running baseline methods with ensemble_size={ensemble_size}")
    print(f"    Baseline input: GT[13:24] shape = {gt_context_np.shape}")
    
    if PYSTEPS_AVAILABLE:
        try:
            # PySTEPS ensemble
            print("    Running PySTEPS ensemble...")
            steps_result = run_pysteps_ensemble(gt_context_np, timesteps, ensemble_size)
            if steps_result is not None:
                baseline_predictions['steps'] = steps_result
                print(f"    PySTEPS ensemble completed. {len(steps_result)} members")
        except Exception as e:
            print(f"    PySTEPS failed: {e}")
        
        try:
            # LINDA ensemble
            print("    Running LINDA ensemble...")
            linda_result = run_linda_ensemble(gt_context_np, timesteps, ensemble_size)
            if linda_result is not None:
                baseline_predictions['linda'] = linda_result
                print(f"    LINDA ensemble completed. {len(linda_result)} members")
        except Exception as e:
            print(f"    LINDA failed: {e}")
    else:
        print("    Skipping PySTEPS and LINDA (PySTEPS not available)")
    
    # Lagrangian ensemble (repeat pre-computed result with small noise for ensemble)
    try:
        print("    Creating Lagrangian ensemble...")
        lagrangian_target = sample_data['langragian_target']  # LP[25:36]
        if torch.is_tensor(lagrangian_target):
            lagrangian_base = lagrangian_target.cpu().numpy()
        else:
            lagrangian_base = lagrangian_target
        
        # Create ensemble by adding small noise to the deterministic result
        lagrangian_ensemble = []
        for i in range(ensemble_size):
            if i == 0:
                # First member is the original
                member = lagrangian_base.copy()
            else:
                # Add small amount of noise for ensemble diversity
                noise_scale = 0.05 * i  # Increasing noise for diversity
                noise = np.random.normal(0, noise_scale, lagrangian_base.shape)
                member = lagrangian_base + noise
                # Ensure non-negative precipitation
                member = np.maximum(member, 0.0)
            lagrangian_ensemble.append(member)
        
        baseline_predictions['lagrangian'] = lagrangian_ensemble
        print(f"    Lagrangian ensemble completed. {len(lagrangian_ensemble)} members")
    except Exception as e:
        print(f"    Lagrangian ensemble failed: {e}")
    
    try:
        # Naive persistence ensemble
        print("    Running Naive persistence ensemble...")
        naive_ensemble = []
        for i in range(ensemble_size):
            # Use last frame as persistence
            last_frame = gt_context_np[-1]  # (H, W)
            if i == 0:
                # First member is pure persistence
                naive_result = np.stack([last_frame] * timesteps, axis=0)  # (T, H, W)
            else:
                # Add noise for ensemble diversity
                noise_scale = 0.1 * i
                noise = np.random.normal(0, noise_scale, (timesteps, last_frame.shape[0], last_frame.shape[1]))
                naive_result = np.stack([last_frame] * timesteps, axis=0) + noise
                # Ensure non-negative precipitation
                naive_result = np.maximum(naive_result, 0.0)
            
            naive_ensemble.append(naive_result)
        
        baseline_predictions['naive'] = naive_ensemble
        print(f"    Naive ensemble completed. {len(naive_ensemble)} members")
    except Exception as e:
        print(f"    Naive ensemble failed: {e}")
    
    return baseline_predictions

def run_pysteps_ensemble(precip_sequence, timesteps=12, n_ens_members=9):
    """Run PySTEPS ensemble nowcasting"""
    if not PYSTEPS_AVAILABLE:
        return None
    
    try:
        # Transform to dB scale
        R_train, _ = transformation.dB_transform(precip_sequence, threshold=0.1, zerovalue=-15.0)
        R_train[~np.isfinite(R_train)] = -15.0
        
        # Estimate motion field
        V = dense_lucaskanade(R_train)
        
        # STEPS nowcast
        nowcast_method = nowcasts.get_method("steps")
        R_forecast = nowcast_method(
            R_train, V, timesteps, 
            n_ens_members=n_ens_members, 
            n_cascade_levels=6,
            precip_thr=-10.0, 
            kmperpixel=10, 
            timestep=30, 
            return_output=True
        )
        
        # Back-transform to rain rates
        R_forecast_linear = transformation.dB_transform(R_forecast, threshold=-10.0, inverse=True)[0]
        
        # Convert to list format - PySTEPS returns (n_ens, timesteps, H, W)
        ensemble_forecasts = [R_forecast_linear[i] for i in range(n_ens_members)]
        
        return ensemble_forecasts
        
    except Exception as e:
        print(f"PySTEPS failed: {e}")
        return None

def run_linda_ensemble(precip_sequence, timesteps=12, n_ens_members=9):
    """Run LINDA ensemble nowcasting"""
    if not PYSTEPS_AVAILABLE:
        return None
    
    try:
        # Estimate motion field
        V = dense_lucaskanade(precip_sequence)
        nowcast_method = nowcasts.get_method("linda")
        
        # LINDA nowcast
        forecast = nowcast_method(
            precip_sequence, V, timesteps, 
            max_num_features=15, 
            add_perturbations=True, 
            n_ens_members=n_ens_members, 
            return_output=True, 
            vel_pert_method=None,
            num_workers=1
        )
        
        # Convert to list format
        ensemble_forecasts = [forecast[i] for i in range(n_ens_members)]
        
        return ensemble_forecasts
        
    except Exception as e:
        print(f"LINDA failed: {e}")
        return None

def calculate_crps_ensemble(observations, forecasts):
    """
    Calculate CRPS for ensemble forecasts.
    
    Args:
        observations: Ground truth values (T, H, W) or similar
        forecasts: Ensemble forecasts (E, T, H, W) or list of (T, H, W)
    
    Returns:
        crps_scores: CRPS scores for each time step (T,)
    """
    if not CRPS_AVAILABLE:
        raise ImportError("properscoring not available for CRPS calculation")
    
    # Convert list of forecasts to array if needed
    if isinstance(forecasts, list):
        forecasts = np.stack(forecasts)  # (E, T, H, W)
    
    # Ensure observations are (T, H, W)
    if len(observations.shape) == 3:
        obs = observations  # (T, H, W)
    elif len(observations.shape) == 4:
        obs = observations[0]  # (B, T, H, W) -> (T, H, W)
    elif len(observations.shape) == 5:
        obs = observations[0, :, 0]  # (B, T, C, H, W) -> (T, H, W)
    else:
        raise ValueError(f"Unexpected observation shape: {observations.shape}")

    # Ensure forecasts are (E, T, H, W)
    if len(forecasts.shape) == 4:
        fcst = forecasts  # (E, T, H, W)
    elif len(forecasts.shape) == 5:
        fcst = forecasts[:, 0]  # (E, B, T, H, W) -> (E, T, H, W)
    elif len(forecasts.shape) == 6:
        fcst = forecasts[:, 0, :, 0]  # (E, B, T, C, H, W) -> (E, T, H, W)
    else:
        raise ValueError(f"Unexpected forecast shape: {forecasts.shape}")

    # Verify shapes match
    if fcst.shape[1:] != obs.shape:
        raise ValueError(f"Shape mismatch: forecasts {fcst.shape[1:]} vs obs {obs.shape}")

    num_timesteps = obs.shape[0]
    crps_scores = []
    
    for t in range(num_timesteps):
        obs_t = obs[t]  # (H, W)
        forecast_t = fcst[:, t]  # (E, H, W)
        
        try:
            # Reshape for vectorized calculation
            obs_flat = obs_t.flatten()  # (H*W,)
            forecast_flat = forecast_t.reshape(forecast_t.shape[0], -1).T  # (H*W, E)
            
            # Calculate CRPS for all spatial points at once
            crps_spatial = ps.crps_ensemble(obs_flat, forecast_flat)  # (H*W,)
            
            # Average over spatial dimensions
            crps_score = np.nanmean(crps_spatial)
            crps_scores.append(crps_score)
            
        except Exception as e:
            print(f"CRPS calculation failed for timestep {t}: {e}")
            crps_scores.append(np.nan)
    
    return np.array(crps_scores)

def calculate_crpss(method_crps, reference_crps, min_reference_crps=1e-8):
    """
    Calculate CRPSS (CRPS Skill Score) relative to reference method.
    CRPSS = 1 - CRPS(method) / CRPS(reference)
    """
    # Convert to numpy arrays
    method_crps = np.asarray(method_crps)
    reference_crps = np.asarray(reference_crps)
    
    # Handle zero or very small reference CRPS values
    safe_reference = np.where(reference_crps < min_reference_crps, min_reference_crps, reference_crps)
    
    # Calculate CRPSS
    crpss = 1 - (method_crps / safe_reference)
    
    # Handle edge cases
    perfect_method_mask = (method_crps < min_reference_crps) & (reference_crps >= min_reference_crps)
    crpss[perfect_method_mask] = 1.0
    
    both_perfect_mask = (method_crps < min_reference_crps) & (reference_crps < min_reference_crps)
    crpss[both_perfect_mask] = 0.0
    
    # Clip to reasonable range
    crpss = np.clip(crpss, -2.0, 1.0)
    
    return crpss

def plot_crps_results(crps_results, save_dir):
    """Plot CRPS results for all methods"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create time axis from -3.5h to +2.0h (12 timesteps)
    time_axis = np.linspace(-3.5, 2.0, 12)
    time_labels = [f"{t:+.1f}h" for t in time_axis]
    
    # Colors for different methods
    colors = {
        'ldm': 'black',
        'steps': 'blue', 
        'linda': 'green',
        'lagrangian': 'orange',
        'naive': 'red'
    }
    
    # Plot 1: CRPS scores
    plt.figure(figsize=(14, 8))
    
    for method, crps_values in crps_results.items():
        if len(crps_values) > 0:
            plt.plot(time_axis, crps_values, 
                    color=colors.get(method, 'black'),
                    marker='o', linewidth=2, markersize=4,
                    label=method.upper())
    
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('CRPS', fontsize=12)
    plt.title('CRPS Scores - LDM vs Baselines (Ensemble Size=9)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(-3.5, 2.0)
    plt.xticks(time_axis, time_labels, rotation=45)
    
    # Save plot
    crps_plot_path = os.path.join(save_dir, 'crps_scores_ensemble.png')
    plt.savefig(crps_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: CRPSS relative to PySTEPS
    if 'steps' in crps_results:
        plt.figure(figsize=(14, 8))
        
        steps_crps = crps_results['steps']
        
        for method, crps_values in crps_results.items():
            if method != 'steps' and len(crps_values) > 0:
                crpss_values = calculate_crpss(crps_values, steps_crps)
                plt.plot(time_axis, crpss_values,
                        color=colors.get(method, 'black'),
                        marker='s', linewidth=2, markersize=4,
                        label=f'{method.upper()} vs STEPS')
        
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        plt.xlabel('Time (hours)', fontsize=12)
        plt.ylabel('CRPSS', fontsize=12)
        plt.title('CRPS Skill Score', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(-3.5, 2.0)
        plt.xticks(time_axis, time_labels, rotation=45)
        
        # Save plot
        crpss_plot_path = os.path.join(save_dir, 'crpss_scores_ensemble.png')
        plt.savefig(crpss_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"CRPS plots saved:")
        print(f"  CRPS scores: {crps_plot_path}")
        print(f"  CRPSS scores: {crpss_plot_path}")

def save_predictions_to_hdf5(predictions_dict, ground_truths, file_path, sample_indices):
    """Save all predictions to HDF5 file"""
    print(f"Saving predictions to: {file_path}")
    
    with h5py.File(file_path, 'w') as f:
        # Save metadata
        f.attrs['num_samples'] = len(sample_indices)
        f.attrs['sample_indices'] = sample_indices
        f.attrs['methods'] = list(predictions_dict.keys())
        f.attrs['ensemble_size'] = len(predictions_dict[list(predictions_dict.keys())[0]][0])
        
        # Save ground truth
        gt_group = f.create_group('ground_truth')
        for i, gt in enumerate(ground_truths):
            gt_group.create_dataset(f'sample_{i:03d}', data=gt)
        
        # Save predictions for each method
        for method_name, predictions in predictions_dict.items():
            method_group = f.create_group(method_name)
            for i, pred_ensemble in enumerate(predictions):
                if pred_ensemble is not None and len(pred_ensemble) > 0:
                    # Stack ensemble members
                    if isinstance(pred_ensemble[0], torch.Tensor):
                        ensemble_array = torch.stack(pred_ensemble).numpy()
                    else:
                        ensemble_array = np.stack(pred_ensemble)
                    method_group.create_dataset(f'sample_{i:03d}', data=ensemble_array)
    
    print(f"  Predictions saved successfully")

def main():
    args = parse_args()
    
    print("="*60)
    print("CRPS/CRPSS CALCULATION FOR LATENT DIFFERENCE LDM")
    print("="*60)
    print(f"Number of samples: {args.num_samples}")
    print(f"DDPM steps: {args.num_inference_steps}")
    print(f"Ensemble size: {args.ensemble_size}")
    print(f"Training paradigm: Latent Difference (GT - Lagrangian)")
    print(f"VAE Decoder: CRPS Fine-tuned Generative (non-deterministic)")
    print(f"CRPS Available: {CRPS_AVAILABLE}")
    print("="*60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    unet, scheduler = load_ldm_model(args.ldm_checkpoint, device)
    vae_model = load_crps_finetuned_vae(args.vae_checkpoint, device)
    
    # Get test data
    test_dataset, test_indices = get_test_data(
        args.vae_checkpoint, device, args.num_samples, args.seed
    )
    
    # Initialize results storage
    all_crps_results = {}
    method_names = ['ldm', 'steps', 'linda']
    
    for method in method_names:
        all_crps_results[method] = []
    
    # Storage for predictions if saving
    if args.save_predictions:
        all_predictions = {method: [] for method in method_names}
        all_ground_truths = []
    
    # Process each sample
    total_samples = len(test_indices)
    print(f"\nProcessing {total_samples} samples...")
    
    for i, idx in enumerate(tqdm(test_indices, desc="Processing samples")):
        print(f"\nSample {i+1}/{total_samples}")
        
        # Get sample
        sample = test_dataset[idx]
        for key in sample:
            if torch.is_tensor(sample[key]):
                sample[key] = sample[key].unsqueeze(0)
        
        # Ground truth - use gt_target (GT[25:36]) from LangragianLDMDataset
        ground_truth = sample['gt_target']  # GT[25:36] - the target period
        if torch.is_tensor(ground_truth):
            gt_numpy = ground_truth.cpu().numpy()
        else:
            gt_numpy = ground_truth
        
        # Ensure proper shape for CRPS calculation (T, H, W)
        if len(gt_numpy.shape) == 4:  # (B, T, H, W)
            gt_numpy = gt_numpy[0, :, :, :]
        elif len(gt_numpy.shape) == 5:  # (B, T, C, H, W)
            gt_numpy = gt_numpy[0, :, 0, :, :]
        
        if args.save_predictions:
            all_ground_truths.append(gt_numpy)
        
        # LDM ensemble predictions
        print("  Running LDM ensemble...")
        ldm_ensemble = generate_ldm_ensemble_predictions(
            unet, scheduler, vae_model, test_dataset.dataset, sample, device, 
            args.num_inference_steps, args.ensemble_size
        )
        
        # Convert LDM ensemble to numpy
        ldm_ensemble_np = []
        for pred in ldm_ensemble:
            pred_np = pred.detach().cpu().numpy()
            if len(pred_np.shape) == 5:  # (B, T, C, H, W)
                pred_np = pred_np[0, :, 0, :, :]  # (T, H, W)
            ldm_ensemble_np.append(pred_np)
        
        if args.save_predictions:
            all_predictions['ldm'].append(ldm_ensemble_np)
        
        # Baseline ensemble predictions
        print("  Running baseline ensembles...")
        baseline_ensembles = run_baseline_ensemble_methods(sample, args.ensemble_size)
        
        # Save baseline predictions (only for methods in our list)
        if args.save_predictions:
            for method_name in method_names:
                if method_name != 'ldm':  # Skip LDM, already saved above
                    if method_name in baseline_ensembles:
                        all_predictions[method_name].append(baseline_ensembles[method_name])
                    else:
                        all_predictions[method_name].append(None)
        
        # Calculate CRPS for LDM
        if CRPS_AVAILABLE:
            ldm_crps = calculate_crps_ensemble(gt_numpy, ldm_ensemble_np)
            all_crps_results['ldm'].append(ldm_crps)
            
            # Calculate CRPS for baselines (only for methods in our list)
            for method_name, ensemble_preds in baseline_ensembles.items():
                if method_name in method_names:  # Only process methods we're tracking
                    if ensemble_preds is not None:
                        baseline_crps = calculate_crps_ensemble(gt_numpy, ensemble_preds)
                        all_crps_results[method_name].append(baseline_crps)
                    else:
                        all_crps_results[method_name].append(np.full(12, np.nan))
    
    # Save predictions if requested
    if args.save_predictions:
        predictions_file = os.path.join(args.output_dir, args.predictions_file)
        save_predictions_to_hdf5(all_predictions, all_ground_truths, predictions_file, test_indices)
    
    # Calculate average CRPS results
    if CRPS_AVAILABLE:
        print("\n" + "="*60)
        print("FINAL CRPS RESULTS (LATENT DIFFERENCE LDM)")
        print("="*60)
        
        averaged_crps_results = {}
        for method in method_names:
            if all_crps_results[method]:
                # Filter out NaN arrays and average
                valid_results = [crps for crps in all_crps_results[method] if not np.all(np.isnan(crps))]
                if valid_results:
                    stacked = np.stack(valid_results, axis=0)
                    averaged_crps_results[method] = np.nanmean(stacked, axis=0)
                    
                    # Print average CRPS for this method
                    avg_crps = np.nanmean(averaged_crps_results[method])
                    print(f"{method.upper()}: Average CRPS = {avg_crps:.6f}")
        
        # Plot CRPS results
        if averaged_crps_results:
            plot_crps_results(averaged_crps_results, args.output_dir)
        
        print(f"\nAnalysis completed! Results saved to {args.output_dir}")
        print("Generated plots:")
        print("  - CRPS scores comparison (ensemble size=9)")
        print("  - CRPSS scores relative to PySTEPS")
        if args.save_predictions:
            predictions_file = os.path.join(args.output_dir, args.predictions_file)
            print(f"  - Ensemble predictions saved to: {predictions_file}")
    else:
        print("CRPS calculation not available - install properscoring package")

if __name__ == "__main__":
    main()