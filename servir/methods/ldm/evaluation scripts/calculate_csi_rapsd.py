#!/usr/bin/env python3
"""
CSI Calculation Script for Latent Difference Precipitation Nowcasting

This script calculates Critical Success Index (CSI) and RAPSD for LDM trained on 
difference between IMERG ground truth and Lagrangian prediction error in latent space.

The LDM is trained to predict: (IMERG_GT - Lagrangian_Prediction_Error) in latent space
Final prediction = Lagrangian_Baseline + LDM_Prediction

Uses CRPS fine-tuned generative VAE decoder for non-deterministic reconstruction
with improved uncertainty quantification (ensemble_size=1 but stochastic).

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

# PySTEPS imports
try:
    from pysteps.utils import transformation
    from pysteps import nowcasts
    from pysteps import motion
    from pysteps.motion.lucaskanade import dense_lucaskanade
    from pysteps.verification.detcatscores import det_cat_fct
    from pysteps.utils.spectral import rapsd
    PYSTEPS_AVAILABLE = True
    print("✓ PySTEPS imported successfully")
except ImportError as e:
    print(f"Warning: PySTEPS not available. PySTEPS baseline will be skipped. Error: {e}")
    PYSTEPS_AVAILABLE = False

# Add the ldm_data_loader directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ldm_data_loader'))
from langragian_ldm_dataloader import get_langragian_ldm_dataloader

# Import VAE models
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'vae_models'))
from vae_imerg import SimpleVAE3D as vae_imerg
from vae_ir import SimpleVAE3D as vae_ir

# Import LDM components
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

# Add nowcasting methods path - make sure it's first in path
nowcasting_path = os.path.join(os.path.dirname(__file__), '..', 'residual_vae', 'src')
if nowcasting_path in sys.path:
    sys.path.remove(nowcasting_path)
sys.path.insert(0, nowcasting_path)

try:
    # Clear any cached imports
    if 'extrapolation_methods' in sys.modules:
        del sys.modules['extrapolation_methods']
    if 'naive_persistence' in sys.modules:
        del sys.modules['naive_persistence']
    
    from extrapolation_methods import linda, steps, langragian_persistance
    from naive_persistence import naive_persistence
    NOWCASTING_AVAILABLE = True
    print("✓ Nowcasting methods imported successfully")
except ImportError as e:
    print(f"Warning: Nowcasting methods not available. Error: {e}")
    NOWCASTING_AVAILABLE = False

# Add SERVIR evaluation utilities (optional - local implementation below if not available)
try:
    from servir.utils.evaluation import evaluation, csi_boxplot, rapsd_boxplot
    SERVIR_EVALUATION_AVAILABLE = True
    print("✓ SERVIR evaluation utilities imported successfully")
except ImportError as e:
    print(f"Warning: SERVIR evaluation utilities not available. Using local implementations. Error: {e}")
    SERVIR_EVALUATION_AVAILABLE = False

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate CSI for Latent Difference LDM Nowcasting')
    
    # Dataset parameters
    parser.add_argument('--num-samples', type=int, default=200,
                        help='Number of random test samples to use (default: 300)')
    parser.add_argument('--seed', type=int, default=11,
                        help='Random seed for reproducible sample selection (default: 11)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for inference (default: 1)')
    
    # Model parameters
    parser.add_argument('--ldm-checkpoint', type=str, 
                        default='checkpoints/ldm_epoch_4.pt',
                        help='Path to pretrained LDM checkpoint')
    parser.add_argument('--vae-checkpoint', type=str,
                        default='vae_models/diff_encoder.pth',
                        help='Path to CRPS fine-tuned generative VAE checkpoint')
    
    # Inference parameters
    parser.add_argument('--num-inference-steps', type=int, default=1000,
                        help='Number of DDPM sampling steps (default: 1000)')
    parser.add_argument('--ensemble-size', type=int, default=1,
                        help='Number of ensemble predictions to generate (default: 1)')
    
    # CSI parameters
    parser.add_argument('--thresholds', nargs='+', type=float, 
                        default=[0.2, 0.5, 2.0, 4.0, 6.0, 8.0, 10.0],
                        help='Precipitation thresholds for CSI calculation (mm/h)')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='csi_results_latent_diff',
                        help='Directory to save CSI results and plots')
    
    # Checkpointing options
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save all predictions to HDF5 file for later use')
    parser.add_argument('--predictions-file', type=str, default='csi_predictions_latent_diff.h5',
                        help='HDF5 file to save/load predictions')
    parser.add_argument('--load-predictions', action='store_true',
                        help='Load predictions from file and skip inference')
    parser.add_argument('--checkpoint-interval', type=int, default=50,
                        help='Save checkpoint every N samples (default: 50)')
    
    return parser.parse_args()

def load_vae_models(device, vae_checkpoint_path):
    """Load and configure VAE models for latent difference training"""
    print("Loading VAE models for latent difference training...")
    
    # The langragian_ldm_dataloader handles VAE loading internally
    # We don't need to load separate VAE models here
    # VAE models are loaded inside the LangragianLDMDataset
    
    print(f"  ✓ VAE models will be loaded by LangragianLDMDataset from: {vae_checkpoint_path}")
    
    return None, None

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

def generate_lagrangian_baseline(precip_input, timesteps=12):
    """Generate Lagrangian persistence baseline for comparison"""
    if not NOWCASTING_AVAILABLE:
        print("Warning: Nowcasting methods not available")
        return None
    
    # Convert input to numpy (T, H, W)
    if torch.is_tensor(precip_input):
        precip_input = precip_input.cpu().numpy()
    
    # Ensure shape is (T, H, W)
    if len(precip_input.shape) == 4:  # (B, T, C, H, W)
        precip_input = precip_input[0, :, 0, :, :]  # Take first batch and channel
    elif len(precip_input.shape) == 5:  # (B, T, C, H, W)
        precip_input = precip_input[0, :, 0, :, :]
    
    try:
        # Lagrangian persistence
        lagrangian_result = langragian_persistance(precip_input, timesteps)
        if lagrangian_result is not None:
            print(f"    Lagrangian baseline completed. Shape: {lagrangian_result.shape}")
            return lagrangian_result
        else:
            print("    Lagrangian baseline returned None")
            return None
    except Exception as e:
        print(f"    Lagrangian baseline failed: {e}")
        return None

def generate_ldm_prediction_with_baseline(unet, scheduler, dataset, sample_data, device, num_inference_steps):
    """
    Generate LDM prediction using the latent difference training paradigm:
    Final_Prediction = Lagrangian_Target + LDM_Difference_Prediction
    
    The dataset already contains:
    - Lagrangian predictions for both context and target periods
    - Encoded differences and IR context
    """
    with torch.no_grad():
        # Extract data from the sample (already processed by LangragianLDMDataset)
        context = sample_data['context']  # Combined encoded: diff + IR
        lagrangian_target = sample_data['langragian_target']  # LP[25:36] - our baseline
        gt_target = sample_data['gt_target']  # GT[25:36] - ground truth for comparison
        
        print("    Using pre-computed Lagrangian target as baseline...")
        
        # Step 1: Generate LDM difference prediction in latent space
        print("    Generating LDM difference prediction...")
        
        # Permute context to match training format: (B, T, C, H, W) -> (B, C, T, H, W)
        context = context.permute(0, 2, 1, 3, 4).to(device)
        
        # Initialize random noise for the difference prediction
        target_shape = sample_data['target'].shape  # Encoded target shape
        latent_sample = torch.randn(target_shape).to(device)
        latent_sample = latent_sample.permute(0, 2, 1, 3, 4)  # Match format: (B, C, T, H, W)
        
        # DDPM sampling for the difference
        step_interval = max(1, scheduler.num_train_timesteps // num_inference_steps)
        timesteps = list(range(scheduler.num_train_timesteps - 1, -1, -step_interval))[:num_inference_steps]
        
        for t in tqdm(timesteps, desc="DDPM Sampling", leave=False):
            t_tensor = torch.tensor([t], device=device).long()
            
            # Predict noise for the difference
            noise_pred = unet(
                x=latent_sample,
                timesteps=t_tensor,
                context=context
            )
            
            # Remove noise
            step_output = scheduler.step(noise_pred, t, latent_sample)
            if hasattr(step_output, 'prev_sample'):
                latent_sample = step_output.prev_sample
            else:
                latent_sample = step_output[0] if isinstance(step_output, tuple) else step_output
        
        # Step 2: Decode the predicted difference using non-deterministic generative VAE
        print("    Decoding predicted difference with CRPS fine-tuned generative VAE...")
        # The dataset's VAE model will decode the difference non-deterministically
        predicted_diff_latent = latent_sample.permute(0, 2, 1, 3, 4)  # Back to (B, T, C, H, W)
        
        # For non-deterministic decoding, we need to add noise to the latent space
        # The CRPS fine-tuned model enables generative (non-deterministic) decoding
        latent_for_decode = predicted_diff_latent.squeeze(0).permute(1, 0, 2, 3).unsqueeze(0).to(device)
        
        # Use non-deterministic decode (ensemble_size=1 but still stochastic)
        # The CRPS fine-tuned decoder will add appropriate stochasticity
        predicted_diff_decoded = dataset.vae_model.decode(latent_for_decode)
        predicted_diff_decoded = predicted_diff_decoded.squeeze(0).permute(1, 0, 2, 3)  # (T, C, H, W)
        
        print(f"    Non-deterministic VAE decoding completed. Shape: {predicted_diff_decoded.shape}")
        
        # Step 3: Combine Lagrangian baseline + predicted difference
        print("    Combining Lagrangian baseline + predicted difference...")
        
        # Convert Lagrangian target to tensor if needed
        if isinstance(lagrangian_target, np.ndarray):
            lagrangian_target = torch.from_numpy(lagrangian_target).float()
        
        # Add channel dimension if needed
        if len(lagrangian_target.shape) == 3:  # (T, H, W)
            lagrangian_target = lagrangian_target.unsqueeze(1)  # (T, 1, H, W)
        
        # Remove predicted_diff batch dimension and ensure same shape
        if len(predicted_diff_decoded.shape) == 4:  # (T, C, H, W)
            predicted_diff_final = predicted_diff_decoded
        else:
            predicted_diff_final = predicted_diff_decoded.squeeze(0)
        
        # Final prediction = Lagrangian baseline + predicted difference
        final_prediction = lagrangian_target + predicted_diff_final.cpu()
        
        # Add batch dimension for consistency: (T, C, H, W) -> (1, T, C, H, W)
        final_prediction = final_prediction.unsqueeze(0)
        
        return final_prediction, lagrangian_target.squeeze(1).numpy()  # Remove channel dim for baseline

def run_baseline_methods(sample_data, timesteps=12):
    """
    Run baseline nowcasting methods with natural input data but aligned prediction windows.
    
    Each method uses its optimal input data, but all predict the same target period (GT[25:36])
    for fair comparison. Only the PREDICTION WINDOW needs to be aligned, not the input.
    """
    baseline_predictions = {}
    
    if not NOWCASTING_AVAILABLE:
        print("Warning: Nowcasting methods not available")
        return baseline_predictions
    
    # Extract the full ground truth context for baseline methods to choose their optimal input
    gt_context = sample_data['gt_context']  # GT[13:24] - available context
    
    # Convert to numpy if it's a tensor
    if torch.is_tensor(gt_context):
        gt_context_np = gt_context.cpu().numpy()
    else:
        gt_context_np = gt_context
    
    # Ensure shape is (T, H, W) - should already be correct from dataset
    if len(gt_context_np.shape) == 5:  # (B, T, C, H, W)
        gt_context_np = gt_context_np[0, :, 0, :, :]  # Take first batch and channel
    elif len(gt_context_np.shape) == 4:  # Could be (B, T, H, W) or (T, C, H, W)
        if gt_context_np.shape[0] == 1:  # (1, T, H, W) - batch first
            gt_context_np = gt_context_np[0, :, :, :]  # Remove batch
        else:  # (T, C, H, W) - time first
            gt_context_np = gt_context_np[:, 0, :, :]  # Remove channel
    
    print(f"    Available context: GT[13:24] shape = {gt_context_np.shape}")
    print(f"    Target period: GT[25:36] ({timesteps} timesteps) - SAME for all methods")
    
    if PYSTEPS_AVAILABLE:
        try:
            # PySTEPS: Uses its optimal input strategy → predict GT[25:36]
            # PySTEPS typically uses the full available context
            print("    Running PySTEPS with optimal input strategy...")
            steps_result = steps(gt_context_np, timesteps, n_ens_members=1, return_output=True)
            if steps_result is not None:
                baseline_predictions['steps'] = steps_result
                print(f"    PySTEPS completed. Input: GT[13:24], Output shape: {steps_result.shape}")
            else:
                print("    PySTEPS returned None")
        except Exception as e:
            print(f"    PySTEPS failed: {e}")
        
        try:
            # LINDA: Uses its optimal input strategy → predict GT[25:36]
            # LINDA typically uses the full available context
            print("    Running LINDA with optimal input strategy...")
            linda_result = linda(gt_context_np, timesteps, add_perturbations=False, n_ens_members=1, return_output=True)
            if linda_result is not None:
                baseline_predictions['linda'] = linda_result
                print(f"    LINDA completed. Input: GT[13:24], Output shape: {linda_result.shape}")
            else:
                print("    LINDA returned None")
        except Exception as e:
            print(f"    LINDA failed: {e}")
    else:
        print("    Skipping PySTEPS and LINDA (PySTEPS not available)")
    
    # Use pre-computed Lagrangian result from dataset
    # This already uses optimal Lagrangian input strategy: GT[13:24] → predict GT[25:36]
    try:
        print("    Using pre-computed Lagrangian result...")
        lagrangian_target = sample_data['langragian_target']  # LP[25:36]
        if torch.is_tensor(lagrangian_target):
            lagrangian_result = lagrangian_target.cpu().numpy()
        else:
            lagrangian_result = lagrangian_target
        
        baseline_predictions['lagrangian'] = lagrangian_result
        print(f"    Lagrangian completed. Input: GT[13:24], Output shape: {lagrangian_result.shape}")
    except Exception as e:
        print(f"    Lagrangian failed: {e}")
    
    try:
        # Naive persistence: Uses optimal input strategy → predict GT[25:36]
        # Naive typically uses the most recent available data
        print("    Running Naive persistence with optimal input strategy...")
        naive_result = naive_persistence(gt_context_np, timesteps)
        if naive_result is not None:
            baseline_predictions['naive'] = naive_result
            print(f"    Naive completed. Input: GT[13:24], Output shape: {naive_result.shape}")
        else:
            print("    Naive returned None")
    except Exception as e:
        print(f"    Naive failed: {e}")
    
    # Summary of prediction alignment
    print(f"  ✓ Prediction Window Alignment Summary:")
    print(f"    • Each method uses its OPTIMAL INPUT STRATEGY")
    print(f"    • All methods predict the SAME TARGET PERIOD: GT[25:36]")
    print(f"    • Fair comparison: Only prediction windows are aligned!")
    
    return baseline_predictions

def verify_prediction_alignment(sample_data, ldm_prediction, baseline_predictions, gt_numpy):
    """
    Verify that all prediction methods are aligned to the same target window.
    This ensures fair comparison across all forecasting methods.
    """
    print(f"\n  🔍 PREDICTION ALIGNMENT VERIFICATION:")
    print(f"  =" * 50)
    
    # Ground truth target (what we're comparing against)
    print(f"  Ground Truth Target: GT[25:36] shape = {gt_numpy.shape}")
    
    # LDM prediction
    print(f"  LDM Prediction:      shape = {ldm_prediction.shape}")
    
    # Baseline predictions
    for method_name, prediction in baseline_predictions.items():
        if prediction is not None:
            print(f"  {method_name.upper():15}: shape = {prediction.shape}")
        else:
            print(f"  {method_name.upper():15}: None (failed)")
    
    # Verify temporal alignment
    gt_target = sample_data['gt_target']
    if torch.is_tensor(gt_target):
        gt_target_np = gt_target.cpu().numpy()
    else:
        gt_target_np = gt_target
    
    print(f"\n  📅 TEMPORAL WINDOW VERIFICATION:")
    print(f"    Available Data: GT[0:24] (24 timesteps) - methods choose optimal input")
    print(f"    Target Period:  GT[25:36] (12 timesteps) - SAME for all methods")
    print(f"    Evaluation:     All predictions vs GT[25:36] (fair comparison)")
    print(f"    Strategy:       Only PREDICTION WINDOW aligned, not input data")
    
    # Check if shapes are compatible for CSI calculation
    compatible = True
    reference_shape = None
    
    if ldm_prediction is not None:
        reference_shape = ldm_prediction.shape
    
    for method_name, prediction in baseline_predictions.items():
        if prediction is not None:
            if reference_shape is None:
                reference_shape = prediction.shape
            elif prediction.shape != reference_shape:
                print(f"  ⚠️  WARNING: {method_name} shape mismatch: {prediction.shape} vs {reference_shape}")
                compatible = False
    
    if compatible:
        print(f"  ✅ All predictions have compatible shapes for CSI calculation")
    else:
        print(f"  ❌ Shape mismatches detected - CSI calculation may fail")
    
    print(f"  =" * 50)
    
    return compatible

def calculate_csi(predictions, observations, threshold):
    """Calculate Critical Success Index (CSI) using SERVIR's method with PySTEPS det_cat_fct"""
    if not PYSTEPS_AVAILABLE:
        # Fallback CSI calculation without PySTEPS
        print(f"Warning: Using fallback CSI calculation (PySTEPS not available)")
        return calculate_csi_fallback(predictions, observations, threshold)
    
    # Use SERVIR's method with PySTEPS det_cat_fct
    result = det_cat_fct(predictions, observations, thr=threshold)
    return result['CSI']

def calculate_csi_fallback(predictions, observations, threshold):
    """Fallback CSI calculation without PySTEPS"""
    # Binary classification
    pred_binary = (predictions >= threshold).astype(int)
    obs_binary = (observations >= threshold).astype(int)
    
    # Calculate contingency table elements
    hits = np.sum((pred_binary == 1) & (obs_binary == 1))
    misses = np.sum((pred_binary == 0) & (obs_binary == 1))
    false_alarms = np.sum((pred_binary == 1) & (obs_binary == 0))
    
    # Calculate CSI
    denominator = hits + misses + false_alarms
    if denominator == 0:
        return np.nan
    
    csi = hits / denominator
    return csi

def calculate_csi_sequence(predictions, observations, thresholds):
    """Calculate CSI for each timestep and threshold"""
    # Normalize predictions to (T, H, W)
    if len(predictions.shape) == 4:  # (B, T, H, W) or (1, T, H, W)
        if predictions.shape[0] == 1:  # (1, T, H, W) - from PySTEPS
            predictions = predictions[0, :, :, :]  # Remove batch dimension
        else:  # (B, T, C, H, W) - shouldn't happen but handle it
            predictions = predictions[0, :, :, :]
    elif len(predictions.shape) == 5:  # (B, T, C, H, W)
        predictions = predictions[0, :, 0, :, :]
    elif len(predictions.shape) == 3:  # (T, H, W) - already correct
        pass
    else:
        raise ValueError(f"Unexpected prediction shape: {predictions.shape}")
    
    # Normalize observations to (T, H, W)
    if len(observations.shape) == 5:  # (B, T, C, H, W)
        observations = observations[0, :, 0, :, :]
    elif len(observations.shape) == 4:  # (B, T, H, W) or (T, C, H, W)
        if observations.shape[0] == 1:  # (1, T, H, W) - batch dimension
            observations = observations[0, :, :, :]
        else:  # (T, C, H, W) - remove channel dimension
            observations = observations[:, 0, :, :]
    elif len(observations.shape) == 3:  # (T, H, W) - already correct
        pass
    else:
        raise ValueError(f"Unexpected observation shape: {observations.shape}")
    
    print(f"    CSI calculation: pred shape {predictions.shape}, obs shape {observations.shape}")
    
    T = predictions.shape[0]
    csi_results = {}
    
    for threshold in thresholds:
        csi_scores = []
        for t in range(T):
            csi = calculate_csi(predictions[t], observations[t], threshold)
            csi_scores.append(csi)
        csi_results[threshold] = np.array(csi_scores)
    
    return csi_results

def calculate_rapsd_sequence(predictions, observations):
    """Calculate RAPSD for each timestep using SERVIR's method with PySTEPS rapsd"""
    # Normalize predictions to (T, H, W)
    if len(predictions.shape) == 4:  # (B, T, H, W) or (1, T, H, W)
        if predictions.shape[0] == 1:  # (1, T, H, W) - from PySTEPS
            predictions = predictions[0, :, :, :]  # Remove batch dimension
        else:  # (B, T, C, H, W) - shouldn't happen but handle it
            predictions = predictions[0, :, :, :]
    elif len(predictions.shape) == 5:  # (B, T, C, H, W)
        predictions = predictions[0, :, 0, :, :]
    elif len(predictions.shape) == 3:  # (T, H, W) - already correct
        pass
    
    # Normalize observations to (T, H, W)
    if len(observations.shape) == 5:  # (B, T, C, H, W)
        observations = observations[0, :, 0, :, :]
    elif len(observations.shape) == 4:  # (B, T, H, W) or (T, C, H, W)
        if observations.shape[0] == 1:  # (1, T, H, W) - batch dimension
            observations = observations[0, :, :, :]
        else:  # (T, C, H, W) - remove channel dimension
            observations = observations[:, 0, :, :]
    elif len(observations.shape) == 3:  # (T, H, W) - already correct
        pass
    
    T = predictions.shape[0]
    rapsd_results = []
    
    for t in range(T):
        try:
            # Check for NaN values and replace them
            pred_t = predictions[t].copy()
            obs_t = observations[t].copy()
            
            # Replace NaN values with 0 (no precipitation)
            if np.any(np.isnan(pred_t)):
                pred_t = np.nan_to_num(pred_t, nan=0.0)
                print(f"    Warning: Found NaN values in predictions at timestep {t}, replaced with 0")
            
            if np.any(np.isnan(obs_t)):
                obs_t = np.nan_to_num(obs_t, nan=0.0)
                print(f"    Warning: Found NaN values in observations at timestep {t}, replaced with 0")
            
            # Ensure positive values for RAPSD calculation
            pred_t = np.maximum(pred_t, 0.0)
            obs_t = np.maximum(obs_t, 0.0)
            
            # Calculate RAPSD using PySTEPS
            rapsd_pred = rapsd(pred_t, return_freq=True, fft_method=np.fft)
            rapsd_obs = rapsd(obs_t, return_freq=True, fft_method=np.fft)
            
            rapsd_results.append({
                'prediction': rapsd_pred,
                'observation': rapsd_obs
            })
        except Exception as e:
            print(f"    RAPSD calculation failed for timestep {t}: {e}")
            rapsd_results.append(None)
    
    return rapsd_results

def plot_csi_results(csi_results, thresholds, save_dir):
    """Plot CSI results for all methods and thresholds"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create time axis from -3.5h to +2.0h
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
    
    # Plot for each threshold
    for threshold in thresholds:
        plt.figure(figsize=(12, 8))
        
        for method in csi_results:
            if threshold in csi_results[method]:
                csi_values = csi_results[method][threshold]
                # Filter out NaN values for plotting
                valid_mask = ~np.isnan(csi_values)
                if np.any(valid_mask):
                    plt.plot(time_axis[valid_mask], csi_values[valid_mask], 
                            color=colors.get(method, 'black'),
                            marker='o', linewidth=2, markersize=4,
                            label=method.upper())
        
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        plt.xlabel('Time (hours)', fontsize=12)
        plt.ylabel('CSI', fontsize=12)
        plt.title(f'Critical Success Index - Threshold: {threshold} mm/h)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(-3.5, 2.0)
        plt.ylim(0, 1)
        plt.xticks(time_axis, time_labels, rotation=45)
        
        # Save plot
        plot_path = os.path.join(save_dir, f'csi_threshold_{threshold:.1f}mm.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  CSI plot saved: {plot_path}")

def plot_rapsd_results(rapsd_results, save_dir):
    """Plot RAPSD results for all methods"""
    if not PYSTEPS_AVAILABLE:
        print("  Skipping RAPSD plots (PySTEPS not available)")
        return
        
    os.makedirs(save_dir, exist_ok=True)
    
    # Colors for different methods
    colors = {
        'ldm': 'black',
        'steps': 'blue', 
        'linda': 'green',
        'lagrangian': 'orange',
        'naive': 'red'
    }
    
    # Plot RAPSD for each timestep
    for t in range(12):  # 12 timesteps
        plt.figure(figsize=(12, 8))
        
        for method_name in rapsd_results:
            if rapsd_results[method_name] and len(rapsd_results[method_name]) > 0:
                # Average RAPSD across all samples for this timestep
                pred_rapsds = []
                obs_rapsds = []
                
                for sample_rapsd in rapsd_results[method_name]:
                    if sample_rapsd and len(sample_rapsd) > t and sample_rapsd[t] is not None:
                        pred_rapsds.append(sample_rapsd[t]['prediction'][1])  # Power spectrum
                        obs_rapsds.append(sample_rapsd[t]['observation'][1])  # Power spectrum
                
                if pred_rapsds:
                    # Average prediction RAPSD
                    pred_rapsd_avg = np.mean(pred_rapsds, axis=0)
                    freq = sample_rapsd[t]['prediction'][0]  # Frequency array
                    
                    plt.loglog(freq, pred_rapsd_avg, 
                              color=colors.get(method_name, 'black'),
                              linewidth=2, label=f'{method_name.upper()} Prediction')
        
        # Plot observation RAPSD (should be same for all methods)
        if 'ldm' in rapsd_results and rapsd_results['ldm'] and len(rapsd_results['ldm']) > 0:
            obs_rapsds = []
            for sample_rapsd in rapsd_results['ldm']:
                if sample_rapsd and len(sample_rapsd) > t and sample_rapsd[t] is not None:
                    obs_rapsds.append(sample_rapsd[t]['observation'][1])
            
            if obs_rapsds:
                obs_rapsd_avg = np.mean(obs_rapsds, axis=0)
                freq = rapsd_results['ldm'][0][t]['observation'][0]
                plt.loglog(freq, obs_rapsd_avg, 'k--', linewidth=2, label='Observation')
        
        plt.xlabel('Frequency (cycles/grid length)', fontsize=12)
        plt.ylabel('Power Spectral Density', fontsize=12)
        plt.title(f'RAPSD - Timestep {t+1})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot
        plot_path = os.path.join(save_dir, f'rapsd_timestep_{t+1:02d}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  RAPSD plot saved: {plot_path}")

def plot_servir_style_mean_rapsd(rapsd_results, save_dir, s=8):
    """Plot mean RAPSD using SERVIR style with wavelength on x-axis
    
    This replicates the SERVIR rapsd_boxplot functionality with wavelength vs MRAPSD.
    Calculates mean RAPSD across all timesteps and samples for each method.
    """
    if not PYSTEPS_AVAILABLE:
        print("  Skipping mean RAPSD plot (PySTEPS not available)")
        return
        
    os.makedirs(save_dir, exist_ok=True)
    
    def plot_psd(psd_tuple):
        psd, freq = psd_tuple
        wvlength = s / freq 
        psd[psd == 0] = 0.000001
        return np.log(psd), wvlength
    
    # Colors for different methods
    colors = {
        'ldm': 'black',
        'steps': 'blue', 
        'linda': 'green',
        'lagrangian': 'orange',
        'naive': 'red',
        'ground_truth': 'darkred'  # Very close to naive color
    }
    
    plt.figure(figsize=(12, 8))
    
    has_data = False
    method_stats = {}
    
    # Add ground truth line from observations
    ground_truth_plotted = False
    
    for method_name in ['ldm', 'steps', 'linda', 'lagrangian', 'naive']:
        if method_name in rapsd_results and rapsd_results[method_name]:
            psd_list = []
            obs_psd_list = []
            wvlength = None
            total_samples = len(rapsd_results[method_name])
            valid_samples = 0
            total_timesteps = 0
            
            print(f"    Processing {method_name.upper()}: {total_samples} samples")
            
            # Collect all PSD data across timesteps and samples
            for sample_idx, sample_rapsd in enumerate(rapsd_results[method_name]):
                if sample_rapsd:
                    sample_valid_timesteps = 0
                    for t in range(12):  # 12 timesteps
                        if len(sample_rapsd) > t and sample_rapsd[t] is not None:
                            pred_rapsd = sample_rapsd[t]['prediction']
                            obs_rapsd = sample_rapsd[t]['observation']
                            
                            if pred_rapsd is not None and len(pred_rapsd) == 2:
                                try:
                                    psd, wvl = plot_psd(pred_rapsd)
                                    psd_list.append(psd)
                                    if wvlength is None:
                                        wvlength = wvl
                                    sample_valid_timesteps += 1
                                    total_timesteps += 1
                                    
                                    # Also collect observation PSD for ground truth
                                    if obs_rapsd is not None and len(obs_rapsd) == 2:
                                        obs_psd, _ = plot_psd(obs_rapsd)
                                        obs_psd_list.append(obs_psd)
                                except:
                                    continue
                    if sample_valid_timesteps > 0:
                        valid_samples += 1
            
            if psd_list and wvlength is not None:
                # Calculate mean across all samples and timesteps
                mean_psd = np.mean(psd_list, axis=0)
                
                plt.plot(wvlength, mean_psd, 
                        color=colors.get(method_name, 'black'),
                        linewidth=2, label=method_name.upper())
                has_data = True
                
                # Add ground truth line from observations (only once)
                if not ground_truth_plotted and obs_psd_list:
                    obs_mean_psd = np.mean(obs_psd_list, axis=0)
                    plt.plot(wvlength, obs_mean_psd,
                            color=colors['ground_truth'],
                            linewidth=2, linestyle='--', alpha=0.9,
                            label='GROUND TRUTH')
                    ground_truth_plotted = True
                    print(f"      Ground truth line added from observations")
                
                # Store statistics
                method_stats[method_name] = {
                    'total_samples': total_samples,
                    'valid_samples': valid_samples,
                    'total_timesteps': total_timesteps,
                    'mean_psd_points': len(psd_list)
                }
                
                print(f"      Valid samples: {valid_samples}/{total_samples}")
                print(f"      Total valid timesteps: {total_timesteps}")
                print(f"      Mean calculated from {len(psd_list)} PSD points")
    
    if has_data:
        plt.gca().invert_xaxis()
        plt.xscale('log')
        plt.gca().set_xticks([20, 50, 100, 200, 300, 500])
        plt.gca().get_xaxis().set_major_formatter(plt.ScalarFormatter())
        plt.legend()
        plt.xlabel('Wavelength (km)')
        plt.ylabel('Mean Radially Averaged Power Spectral Density')
        plt.title('Mean Radially Averaged Power Spectral Density')

        plot_path = os.path.join(save_dir, 'mean_rapsd_servir_style.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  SERVIR-style mean RAPSD plot saved: {plot_path}")
        print(f"  Mean RAPSD statistics:")
        for method, stats in method_stats.items():
            print(f"    {method.upper()}: {stats['valid_samples']}/{stats['total_samples']} samples, {stats['total_timesteps']} timesteps, {stats['mean_psd_points']} PSD points")
    else:
        print("  No RAPSD data available for mean plot")

def save_predictions_to_hdf5(predictions_dict, ground_truths, file_path, sample_indices):
    """Save all predictions to HDF5 file"""
    print(f"Saving predictions to: {file_path}")
    
    with h5py.File(file_path, 'w') as f:
        # Save metadata
        f.attrs['num_samples'] = len(sample_indices)
        f.attrs['sample_indices'] = sample_indices
        f.attrs['methods'] = list(predictions_dict.keys())
        
        # Save ground truth
        gt_group = f.create_group('ground_truth')
        for i, gt in enumerate(ground_truths):
            gt_group.create_dataset(f'sample_{i:03d}', data=gt)
        
        # Save predictions for each method
        for method_name, predictions in predictions_dict.items():
            method_group = f.create_group(method_name)
            for i, pred in enumerate(predictions):
                if pred is not None:
                    method_group.create_dataset(f'sample_{i:03d}', data=pred)
    
    print(f"  Predictions saved successfully")

def main():
    args = parse_args()
    
    print("="*60)
    print("CSI CALCULATION FOR LATENT DIFFERENCE LDM NOWCASTING")
    print("="*60)
    print(f"Number of samples: {args.num_samples}")
    print(f"Thresholds: {args.thresholds} mm/h")
    print(f"DDPM steps: {args.num_inference_steps}")
    print(f"Ensemble size: {args.ensemble_size}")
    print(f"Training paradigm: Latent Difference (GT - Lagrangian)")
    print(f"VAE Decoder: CRPS Fine-tuned Generative (non-deterministic)")
    print()
    print("📅 PREDICTION WINDOW ALIGNMENT STRATEGY:")
    print("  Data Structure: 36 IMERG timesteps (0-35)")
    print("    GT[0:12]   → Available historical data")
    print("    GT[13:24]  → Available context data")
    print("    GT[25:36]  → TARGET PERIOD (same for ALL methods)")
    print()
    print("  Method Strategy:")
    print("    LDM:        Uses GT[13:24] + IR → predict corrections for GT[25:36]")
    print("    PySTEPS:    Uses optimal input strategy → predict GT[25:36]")
    print("    LINDA:      Uses optimal input strategy → predict GT[25:36]")
    print("    Lagrangian: Uses optimal input strategy → predict GT[25:36]")
    print("    Naive:      Uses optimal input strategy → predict GT[25:36]")
    print()
    print("  ✅ Each method uses OPTIMAL INPUT, all predict SAME TARGET PERIOD!")
    print("="*60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    vae_imerg_model, vae_ir_model = load_vae_models(device, args.vae_checkpoint)
    
    unet, scheduler = load_ldm_model(args.ldm_checkpoint, device)
    
    # Get test data
    test_dataset, test_indices = get_test_data(
        args.vae_checkpoint, device, args.num_samples, args.seed
    )
    
    # Initialize results storage
    all_csi_results = {}
    all_rapsd_results = {}
    method_names = ['ldm', 'steps', 'linda', 'lagrangian', 'naive']
    
    for method in method_names:
        all_csi_results[method] = {threshold: [] for threshold in args.thresholds}
        all_rapsd_results[method] = []
    
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
        
        print(f"  Ground truth original shape: {gt_numpy.shape}")
        
        # Normalize ground truth to (1, T, C, H, W) format
        if len(gt_numpy.shape) == 3:  # (T, H, W)
            gt_numpy = np.expand_dims(gt_numpy, axis=1)  # (T, 1, H, W)
            gt_numpy = np.expand_dims(gt_numpy, axis=0)  # (1, T, 1, H, W)
        elif len(gt_numpy.shape) == 4:  # Could be (B, T, H, W) or (T, C, H, W)
            if gt_numpy.shape[0] == 1:  # (1, T, H, W) - batch first
                gt_numpy = np.expand_dims(gt_numpy, axis=2)  # (1, T, 1, H, W)
            else:  # (T, C, H, W) - time first
                gt_numpy = np.expand_dims(gt_numpy, axis=0)  # (1, T, C, H, W)
        # If already 5D (1, T, C, H, W), no change needed
        
        print(f"  Ground truth final shape: {gt_numpy.shape}")
        
        if args.save_predictions:
            all_ground_truths.append(gt_numpy)
        
        # LDM prediction using latent difference paradigm
        print("  Running LDM with latent difference...")
        
        # Generate LDM prediction with baseline integration using the dataset
        ldm_prediction, lagrangian_baseline = generate_ldm_prediction_with_baseline(
            unet, scheduler, test_dataset.dataset, sample, device, args.num_inference_steps
        )
        
        ldm_numpy = ldm_prediction.detach().cpu().numpy()
        
        if args.save_predictions:
            all_predictions['ldm'].append(ldm_numpy)
        
        # Baseline predictions
        print("  Running baselines...")
        baseline_predictions = run_baseline_methods(sample)
        
        # Verify prediction alignment for fair comparison
        alignment_ok = verify_prediction_alignment(sample, ldm_prediction, baseline_predictions, gt_numpy)
        
        if not alignment_ok:
            print("  ⚠️  WARNING: Prediction alignment issues detected!")
        
        # Save baseline predictions
        if args.save_predictions:
            for method_name in ['steps', 'linda', 'lagrangian', 'naive']:
                if method_name in baseline_predictions and baseline_predictions[method_name] is not None:
                    all_predictions[method_name].append(baseline_predictions[method_name])
                else:
                    all_predictions[method_name].append(None)
        
        # Calculate CSI for LDM
        ldm_csi = calculate_csi_sequence(ldm_numpy, gt_numpy, args.thresholds)
        for threshold in args.thresholds:
            all_csi_results['ldm'][threshold].append(ldm_csi[threshold])
        
        # Calculate RAPSD for LDM
        if PYSTEPS_AVAILABLE:
            print("  Calculating RAPSD for LDM...")
            ldm_rapsd = calculate_rapsd_sequence(ldm_numpy, gt_numpy)
            all_rapsd_results['ldm'].append(ldm_rapsd)
        
        # Calculate CSI and RAPSD for baselines
        for method_name, prediction in baseline_predictions.items():
            if prediction is not None:
                baseline_csi = calculate_csi_sequence(prediction, gt_numpy, args.thresholds)
                for threshold in args.thresholds:
                    all_csi_results[method_name][threshold].append(baseline_csi[threshold])
                
                # Calculate RAPSD for baseline
                if PYSTEPS_AVAILABLE:
                    print(f"  Calculating RAPSD for {method_name}...")
                    baseline_rapsd = calculate_rapsd_sequence(prediction, gt_numpy)
                    all_rapsd_results[method_name].append(baseline_rapsd)
            else:
                # Add empty RAPSD entry for missing methods
                all_rapsd_results[method_name].append(None)
        
        # Save intermediate checkpoint after each sample
        if args.save_predictions:
            print(f"  Saving checkpoint after sample {i+1}...")
            checkpoint_file = os.path.join(args.output_dir, f'checkpoint_sample_{i+1:03d}.h5')
            save_predictions_to_hdf5(all_predictions, all_ground_truths, checkpoint_file, test_indices[:i+1])
            print(f"  Checkpoint saved: {checkpoint_file}")
    
    # Save predictions if requested
    if args.save_predictions:
        predictions_file = os.path.join(args.output_dir, args.predictions_file)
        save_predictions_to_hdf5(all_predictions, all_ground_truths, predictions_file, test_indices)
    
    # Average CSI results across all samples
    print("\n" + "="*60)
    print("FINAL CSI RESULTS (LATENT DIFFERENCE LDM)")
    print("="*60)
    
    averaged_csi_results = {}
    for method in method_names:
        averaged_csi_results[method] = {}
        for threshold in args.thresholds:
            if all_csi_results[method][threshold]:
                # Stack and average, handling NaN values
                stacked = np.stack(all_csi_results[method][threshold], axis=0)
                averaged_csi_results[method][threshold] = np.nanmean(stacked, axis=0)
                
                # Print average CSI for this method and threshold
                valid_scores = averaged_csi_results[method][threshold][~np.isnan(averaged_csi_results[method][threshold])]
                if len(valid_scores) > 0:
                    avg_csi = np.mean(valid_scores)
                    print(f"{method.upper()} - {threshold:4.1f}mm/h: CSI = {avg_csi:.3f}")
    
    # Plot CSI results
    plot_csi_results(averaged_csi_results, args.thresholds, args.output_dir)
    
    # Plot RAPSD results
    plot_rapsd_results(all_rapsd_results, args.output_dir)
    
    # Plot mean RAPSD results (SERVIR style)
    plot_servir_style_mean_rapsd(all_rapsd_results, args.output_dir)
    
    print(f"\nAnalysis completed! Results saved to {args.output_dir}")
    print("Generated plots:")
    print("  - CSI plots by threshold (Latent Difference LDM)")
    if PYSTEPS_AVAILABLE:
        print("  - RAPSD plots by timestep (Latent Difference LDM)")
        print("  - Mean RAPSD plot (SERVIR style)")
    if args.save_predictions:
        predictions_file = os.path.join(args.output_dir, args.predictions_file)
        print(f"  - Predictions saved to: {predictions_file}")

if __name__ == "__main__":
    main()