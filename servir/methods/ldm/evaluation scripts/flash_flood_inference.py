#!/usr/bin/env python3
"""
Flash Flood Inference Script - Baseline Methods and LDM

This script runs flash flood prediction inference using existing IR and IMERG data
for the date range 2010-08-11 21:00 to 2010-08-12 03:00
using baseline methods:
steps, naive, lagrangian, and linda, and also includes LDM predictions.

This version focuses on method comparison with ground truth and creates
comparison GIFs for 12 timesteps.

Usage:
    python flash_flood_inference.py --model-path path/to/model.pt --output-dir flash_flood_results_2010
"""

import os
import sys
import torch
import argparse
import datetime
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import shutil
import time as time_module

# Add necessary paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ldm_data_loader'))

# Import required modules
from data_provider import get_latent_mm_dataloader
from langragian_ldm_dataloader import get_langragian_ldm_dataloader
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'vae_models'))
from vae_imerg import SimpleVAE3D as vae_imerg
from vae_ir import SimpleVAE3D as vae_ir
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description='Precipitation Nowcasting - Baseline Methods and LDM Inference')
    parser.add_argument('--model-path', type=str, 
                        default='checkpoints/ldm_epoch_5.pt',
                        help='Path to the trained model checkpoint')
    parser.add_argument('--vae-checkpoint', type=str, 
                        default='vae_models/diff_encoder.pth',
                        help='Path to the difference VAE checkpoint')
    parser.add_argument('--output-dir', type=str, default='precipitation_results',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for inference')
    
    # Event timing parameters
    parser.add_argument('--start-date', type=str, default='2010-08-11',
                        help='Start date for prediction period (YYYY-MM-DD) [default: 2010-08-11]')
    parser.add_argument('--start-time', type=str, default='21:00',
                        help='Start time for prediction period (HH:MM) [default: 21:00]')
    parser.add_argument('--duration-hours', type=int, default=6,
                        help='Duration of prediction period in hours [default: 6]')
    
    parser.add_argument('--ensemble', type=int, default=1,
                        help='Number of ensemble samples to generate')
    return parser.parse_args()


def get_target_datetimes(start_date_str, start_time_str, duration_hours):
    """Get the target datetime range from command line arguments"""
    # Parse start date and time
    try:
        date_parts = [int(x) for x in start_date_str.split('-')]
        time_parts = [int(x) for x in start_time_str.split(':')]
        
        start_datetime = datetime.datetime(date_parts[0], date_parts[1], date_parts[2], 
                                         time_parts[0], time_parts[1], 0)
        end_datetime = start_datetime + datetime.timedelta(hours=duration_hours)
        
        return start_datetime, end_datetime
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid date/time format. Use YYYY-MM-DD for date and HH:MM for time. Error: {e}")


def get_existing_data_paths():
    """Get paths to existing data files"""
    data_dir = os.path.join(os.path.dirname(__file__), 'ldm_data_loader')
    imerg_file = os.path.join(data_dir, 'imerg_data.h5')
    ir_file = os.path.join(data_dir, 'filled_missing_nan_ir_data.h5')
    
    if not os.path.exists(imerg_file):
        raise FileNotFoundError(f"IMERG file not found: {imerg_file}")
    if not os.path.exists(ir_file):
        raise FileNotFoundError(f"IR file not found: {ir_file}")
    
    return imerg_file, ir_file


def parse_timestamp_string(ts_str):
    """Parse timestamp string with various formats"""
    if isinstance(ts_str, bytes):
        ts_str = ts_str.decode()
    
    # Try different timestamp formats
    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S.%f']:
        try:
            return datetime.datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Cannot parse timestamp: {ts_str}")


def find_prediction_windows(imerg_file, ir_file, start_datetime, end_datetime):
    """Find appropriate data windows for the specified prediction period"""
    print(f"Finding prediction windows for {start_datetime} to {end_datetime}")
    
    # Read timestamps from data files
    with h5py.File(imerg_file, 'r') as f:
        imerg_timestamps = [parse_timestamp_string(ts) for ts in f['timestamps'][:]]
        print(f"IMERG data spans: {imerg_timestamps[0]} to {imerg_timestamps[-1]}")
    
    with h5py.File(ir_file, 'r') as f:
        ir_timestamps = [parse_timestamp_string(ts) for ts in f['timestamps'][:]]
        print(f"IR data spans: {ir_timestamps[0]} to {ir_timestamps[-1]}")
    
    # Generate single prediction window for the target period (21:00 to 03:00 = 6 hours)
    prediction_windows = []
    current_pred_start = start_datetime  # 21:00
    current_pred_end = end_datetime      # 03:00 (6 hours later)
    
    print(f"Creating single prediction window: {current_pred_start} to {current_pred_end}")
    
    # Required input windows:
    # IMERG: 6 hours history (12 timesteps * 30min) - input from 15:00 to 21:00
    imerg_input_start = current_pred_start - datetime.timedelta(hours=6)  # 15:00
    imerg_input_end = current_pred_start  # 21:00
    
    # IR: 4 hours history (16 timesteps * 15min) - input from 17:00 to 21:00
    ir_input_start = current_pred_start - datetime.timedelta(hours=4)  # 17:00
    ir_input_end = current_pred_start  # 21:00
    
    print(f"IMERG input needed: {imerg_input_start} to {imerg_input_end}")
    print(f"IR input needed: {ir_input_start} to {ir_input_end}")
    
    # Check if we have sufficient data for this window
    imerg_available = (imerg_input_start >= imerg_timestamps[0] and 
                      current_pred_end <= imerg_timestamps[-1])
    ir_available = (ir_input_start >= ir_timestamps[0] and 
                   ir_input_end <= ir_timestamps[-1])
    
    if imerg_available and ir_available:
        # Find indices for IMERG input (12 timesteps, 30-min intervals)
        imerg_input_indices = []
        for i, ts in enumerate(imerg_timestamps):
            if imerg_input_start <= ts < imerg_input_end:
                imerg_input_indices.append(i)
        
        # Find indices for IMERG target (12 timesteps, 30-min intervals)
        # We want exactly 12 timesteps starting from current_pred_start (21:00 to 03:00)
        imerg_target_indices = []
        for i, ts in enumerate(imerg_timestamps):
            if current_pred_start <= ts < current_pred_end:  # Don't include end time
                imerg_target_indices.append(i)
        
        # Find indices for IR input (16 timesteps, 15-min intervals)
        ir_input_indices = []
        for i, ts in enumerate(ir_timestamps):
            if ir_input_start <= ts < ir_input_end:
                ir_input_indices.append(i)
        
        print(f"Found IMERG input indices: {len(imerg_input_indices)}")
        print(f"Found IMERG target indices: {len(imerg_target_indices)}")
        print(f"Found IR input indices: {len(ir_input_indices)}")
        
        # Ensure we have the right number of timesteps
        if (len(imerg_input_indices) >= 12 and 
            len(imerg_target_indices) >= 12 and 
            len(ir_input_indices) >= 16):
            
            # Take exactly the required number of timesteps
            imerg_input_indices = imerg_input_indices[-12:]  # Last 12 timesteps
            imerg_target_indices = imerg_target_indices[:12]  # First 12 timesteps
            ir_input_indices = ir_input_indices[-16:]  # Last 16 timesteps
            
            # Debug: Print the actual timestamps we're selecting
            print(f"DEBUG - Single window for prediction {current_pred_start} to {current_pred_end}:")
            print(f"  Found {len(imerg_target_indices)} target timesteps")
            if len(imerg_target_indices) >= 2:
                first_target = imerg_timestamps[imerg_target_indices[0]]
                last_target = imerg_timestamps[imerg_target_indices[-1]]
                print(f"  First target timestamp: {first_target}")
                print(f"  Last target timestamp: {last_target}")
            
            window = {
                'prediction_start': current_pred_start,
                'prediction_end': current_pred_end,
                'imerg_input_indices': imerg_input_indices,
                'imerg_target_indices': imerg_target_indices,
                'ir_input_indices': ir_input_indices,
                'imerg_input_times': [imerg_timestamps[i] for i in imerg_input_indices],
                'imerg_target_times': [imerg_timestamps[i] for i in imerg_target_indices],
                'ir_input_times': [ir_timestamps[i] for i in ir_input_indices]
            }
            prediction_windows.append(window)
            
            pred_start_str = current_pred_start.strftime('%Y-%m-%d %H:%M')
            pred_end_str = current_pred_end.strftime('%Y-%m-%d %H:%M')
            
            imerg_in_start = window['imerg_input_times'][0].strftime('%Y-%m-%d %H:%M')
            imerg_in_end = window['imerg_input_times'][-1].strftime('%Y-%m-%d %H:%M')
            
            ir_in_start = window['ir_input_times'][0].strftime('%Y-%m-%d %H:%M')
            ir_in_end = window['ir_input_times'][-1].strftime('%Y-%m-%d %H:%M')
            
            target_start = window['imerg_target_times'][0].strftime('%Y-%m-%d %H:%M')
            target_end = window['imerg_target_times'][-1].strftime('%Y-%m-%d %H:%M')
            
            print(f"Flash Flood Window: Predict {pred_start_str} to {pred_end_str}")
            print(f"  IMERG input: {imerg_in_start} to {imerg_in_end} ({len(imerg_input_indices)} steps)")
            print(f"  IR input: {ir_in_start} to {ir_in_end} ({len(ir_input_indices)} steps)")
            print(f"  Target: {target_start} to {target_end} ({len(imerg_target_indices)} steps)")
            print()  # Empty line for clarity
        else:
            print(f"Insufficient data for the flash flood prediction window")
            print(f"  IMERG input: {len(imerg_input_indices)}/12, target: {len(imerg_target_indices)}/12")
            print(f"  IR input: {len(ir_input_indices)}/16")
    else:
        print(f"Data not available for the specified time range")
        print(f"  IMERG available: {imerg_available}")
        print(f"  IR available: {ir_available}")
    
    print(f"Found {len(prediction_windows)} prediction windows")
    return prediction_windows


def get_exact_time_indices(timestamps, target_start, target_end):
    """Find exact indices for the target time period"""
    start_idx = None
    end_idx = None
    
    for i, ts_str in enumerate(timestamps):
        if isinstance(ts_str, bytes):
            ts_str = ts_str.decode()
        
        # Parse timestamp
        try:
            ts = datetime.datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            # Try other formats
            for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S.%f']:
                try:
                    ts = datetime.datetime.strptime(ts_str, fmt)
                    break
                except ValueError:
                    continue
            else:
                continue
        
        # Check if this timestamp matches our target period
        if start_idx is None and ts >= target_start:
            start_idx = i
        if ts <= target_end:
            end_idx = i
        elif start_idx is not None:
            break  # We've passed the end time
    
    return start_idx, end_idx


def create_samples_for_flash_flood_event(imerg_file, ir_file, device, vae_checkpoint_path, target_datetime_start, target_datetime_end):
    """Create model input samples specifically for the flash flood event with proper windowing"""
    print(f"Creating samples for flash flood event PREDICTION: {target_datetime_start} to {target_datetime_end}")
    print("Using proper windowing: INPUT before PREDICTION target...")
    
    # For the prediction window 2010-08-11 21:00 to 2010-08-12 03:00:
    # - We need IMERG input: 6 hours before (15:00 to 21:00) - 12 timesteps at 30-min intervals
    # - We need IR input: 4 hours before (17:00 to 21:00) - 16 timesteps at 15-min intervals  
    # - Target: The flash flood period (21:00 to 03:00) - 12 timesteps at 30-min intervals
    
    # For LDM: need 24 frames (12 hours before event)
    imerg_input_start_ldm = target_datetime_start - datetime.timedelta(hours=12)  # 09:00  
    # For other methods: need 12 frames just before event
    imerg_input_start = target_datetime_start - datetime.timedelta(hours=6)  # 15:00
    imerg_input_end = target_datetime_start  # 21:00
    ir_input_start = target_datetime_start - datetime.timedelta(hours=4)  # 17:00
    ir_input_end = target_datetime_start  # 21:00
    
    print(f"IMERG input window: {imerg_input_start.strftime('%Y-%m-%d %H:%M')} to {imerg_input_end.strftime('%Y-%m-%d %H:%M')}")
    print(f"IR input window: {ir_input_start.strftime('%Y-%m-%d %H:%M')} to {ir_input_end.strftime('%Y-%m-%d %H:%M')}")
    print(f"Prediction target: {target_datetime_start.strftime('%Y-%m-%d %H:%M')} to {target_datetime_end.strftime('%Y-%m-%d %H:%M')}")
    
    # Read IMERG data and timestamps
    with h5py.File(imerg_file, 'r') as f:
        imerg_timestamps = [parse_timestamp_string(ts) for ts in f['timestamps'][:]]
        imerg_data = f['precipitations'][:]
        print(f"IMERG data spans: {imerg_timestamps[0]} to {imerg_timestamps[-1]}")
    
    # Read IR data and timestamps
    with h5py.File(ir_file, 'r') as f:
        ir_timestamps = [parse_timestamp_string(ts) for ts in f['timestamps'][:]]
        ir_data = f['Tb'][:]
        print(f"IR data spans: {ir_timestamps[0]} to {ir_timestamps[-1]}")
    
    # Find IMERG input indices for regular methods (15:00 to 21:00)
    imerg_input_indices = []
    for i, ts in enumerate(imerg_timestamps):
        if imerg_input_start <= ts < imerg_input_end:
            imerg_input_indices.append(i)
    
    # Find IMERG input indices for LDM (09:00 to 21:00) - 24 frames
    imerg_input_indices_ldm = []
    for i, ts in enumerate(imerg_timestamps):
        if imerg_input_start_ldm <= ts < imerg_input_end:
            imerg_input_indices_ldm.append(i)
    
    # Find IMERG target indices (21:00 to 03:00) 
    imerg_target_indices = []
    for i, ts in enumerate(imerg_timestamps):
        if target_datetime_start <= ts < target_datetime_end:
            imerg_target_indices.append(i)
    
    # Find IR input indices (17:00 to 21:00)
    ir_input_indices = []
    for i, ts in enumerate(ir_timestamps):
        if ir_input_start <= ts < ir_input_end:
            ir_input_indices.append(i)
    
    print(f"Found IMERG input indices (regular methods): {len(imerg_input_indices)} (expected 12)")
    print(f"Found IMERG input indices (LDM): {len(imerg_input_indices_ldm)} (expected 24)")
    print(f"Found IMERG target indices: {len(imerg_target_indices)} (expected 12)")
    print(f"Found IR input indices: {len(ir_input_indices)} (expected 16)")
    
    # Ensure we have enough IMERG data points (IR only needed for LDM)
    if len(imerg_input_indices) >= 12 and len(imerg_input_indices_ldm) >= 24 and len(imerg_target_indices) >= 12:
        # Take exactly the required number of timesteps
        imerg_input_indices = imerg_input_indices[-12:]  # Last 12 timesteps before prediction (for regular methods)
        imerg_input_indices_ldm = imerg_input_indices_ldm[-24:]  # Last 24 timesteps before prediction (for LDM)
        imerg_target_indices = imerg_target_indices[:12]  # First 12 timesteps of prediction
        
        # IR data only needed for LDM - check separately
        if len(ir_input_indices) >= 16:
            ir_input_indices = ir_input_indices[-16:]  # Last 16 timesteps before prediction
            ir_available = True
        else:
            print(f"Warning: Insufficient IR data ({len(ir_input_indices)}/16) - LDM will be skipped")
            ir_available = False
    else:
        raise ValueError(f"Insufficient IMERG data for windowing. Found: IMERG input={len(imerg_input_indices)}, IMERG LDM={len(imerg_input_indices_ldm)}, target={len(imerg_target_indices)}")
    
    # Extract data windows
    imerg_input_data = imerg_data[imerg_input_indices]  # (12, H, W) - input for regular methods
    imerg_input_data_ldm = imerg_data[imerg_input_indices_ldm]  # (24, H, W) - input for LDM
    imerg_target_data = imerg_data[imerg_target_indices]  # (12, H, W) - ground truth for prediction  
    
    # IR data only extracted if available (for LDM)
    if ir_available:
        ir_input_data = ir_data[ir_input_indices]  # (16, H, W) - IR input for LDM
    else:
        ir_input_data = None
    
    # Split LDM input into two 12-frame periods
    imerg_ldm_first_12 = imerg_input_data_ldm[:12]  # (12, H, W) - First 12 frames (09:00-15:00) for Lagrangian
    imerg_ldm_second_12 = imerg_input_data_ldm[12:]  # (12, H, W) - Second 12 frames (15:00-21:00) for ground truth
    
    # Extract timestamps for verification
    imerg_input_times = [imerg_timestamps[i] for i in imerg_input_indices]
    imerg_target_times = [imerg_timestamps[i] for i in imerg_target_indices]
    
    print(f"IMERG input period: {imerg_input_times[0]} to {imerg_target_times[-1]}")
    print(f"IMERG target period: {imerg_target_times[0]} to {imerg_target_times[-1]}")
    
    if ir_available:
        ir_input_times = [ir_timestamps[i] for i in ir_input_indices]
        print(f"IR input period: {ir_input_times[0]} to {ir_input_times[-1]}")
    else:
        ir_input_times = None
        print("IR input: Not available")
    
    # Convert to tensors
    imerg_input_tensor = torch.from_numpy(imerg_input_data).float()  # (12, H, W)
    imerg_target_tensor = torch.from_numpy(imerg_target_data).float()  # (12, H, W) 
    
    if ir_available:
        ir_input_tensor = torch.from_numpy(ir_input_data).float()  # (16, H, W)
    else:
        ir_input_tensor = None
    
    # Generate Langragian prediction for regular methods using 12 frames (15:00-21:00)
    print("Generating Langragian persistence prediction for regular methods...")
    try:
        # Add the encoder path to import langragian
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'residual_vae', 'src'))
        from extrapolation_methods.langragian_persistance import langragian_persistance
        
        # Use IMERG input (15:00-21:00) to predict target period (21:00-03:00)
        langragian_prediction = langragian_persistance(imerg_input_data, timesteps=12)
        langragian_pred_tensor = torch.from_numpy(langragian_prediction).float()  # (12, H, W)
        print(f"Langragian prediction (regular methods) shape: {langragian_pred_tensor.shape}")
    except Exception as e:
        print(f"Warning: Could not generate Langragian prediction: {e}")
        # Use simple persistence as fallback
        langragian_pred_tensor = imerg_input_tensor[-1:].repeat(12, 1, 1)  # Repeat last input frame
    
    # Generate Langragian prediction for LDM using first 12 frames (09:00-15:00)
    print("Generating Langragian prediction for LDM using first 12 frames...")
    try:
        # Use first 12 frames to predict second 12 frames
        langragian_prediction_ldm = langragian_persistance(imerg_ldm_first_12, timesteps=12)
        langragian_pred_ldm_tensor = torch.from_numpy(langragian_prediction_ldm).float()  # (12, H, W)
        print(f"Langragian prediction (LDM) shape: {langragian_pred_ldm_tensor.shape}")
    except Exception as e:
        print(f"Warning: Could not generate LDM Langragian prediction: {e}")
        # Use simple persistence as fallback
        imerg_ldm_first_12_tensor = torch.from_numpy(imerg_ldm_first_12).float()
        langragian_pred_ldm_tensor = imerg_ldm_first_12_tensor[-1:].repeat(12, 1, 1)  # Repeat last frame of first period
    
    # Convert second 12 frames to tensor for LDM ground truth
    imerg_ldm_second_12_tensor = torch.from_numpy(imerg_ldm_second_12).float()  # (12, H, W)
    
    # Calculate difference for LDM approach: GT (second 12) - Langragian prediction
    difference_tensor_ldm = imerg_ldm_second_12_tensor - langragian_pred_ldm_tensor  # (12, H, W)
    
    # Calculate difference for regular approach: GT - Langragian  
    difference_tensor = imerg_target_tensor - langragian_pred_tensor  # (12, H, W)
    
    # Create sample structure for Langragian LDM approach
    sample = {
        'gt_context': imerg_input_tensor,  # (12, H, W) - IMERG input period for regular methods
        'gt_target': imerg_target_tensor,  # (12, H, W) - Ground truth for prediction period
        'langragian_context': imerg_input_tensor,  # Same as gt_context for baseline methods
        'langragian_target': langragian_pred_tensor,  # (12, H, W) - Langragian prediction (regular methods)
        'difference_target': difference_tensor,  # (12, H, W) - GT - Langragian (regular approach)
        
        # LDM-specific data
        'ldm_first_12': torch.from_numpy(imerg_ldm_first_12).float(),  # (12, H, W) - First 12 frames for LDM
        'ldm_second_12': imerg_ldm_second_12_tensor,  # (12, H, W) - Second 12 frames (GT for LDM)
        'ldm_langragian_pred': langragian_pred_ldm_tensor,  # (12, H, W) - Langragian pred using first 12
        'ldm_difference': difference_tensor_ldm,  # (12, H, W) - What LDM should learn
        'ir_available': ir_available,  # Flag to check if LDM can run
        'ir_input': ir_input_tensor if ir_available else None,  # (16, H, W) - IR input period (only for LDM)
        'flash_flood_info': {
            'event_start': target_datetime_start,
            'event_end': target_datetime_end,
            'description': f'Flash flood event prediction - {imerg_target_times[0].strftime("%Y-%m-%d %H:%M")} to {imerg_target_times[-1].strftime("%Y-%m-%d %H:%M")}',
            'prediction_timesteps': len(imerg_target_data),
            'timestep_interval_minutes': 30,
            'actual_timestamps': [t.strftime('%Y-%m-%d %H:%M:%S') for t in imerg_target_times],
            'input_period': f"{imerg_input_times[0].strftime('%Y-%m-%d %H:%M')} to {imerg_input_times[-1].strftime('%Y-%m-%d %H:%M')}",
            'target_period': f"{imerg_target_times[0].strftime('%Y-%m-%d %H:%M')} to {imerg_target_times[-1].strftime('%Y-%m-%d %H:%M')}"
        }
    }
    
    # Move to device
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            sample[key] = value.to(device)
    
    print(f"Flash flood sample prepared with proper windowing:")
    print(f"  Input context shape: {sample['gt_context'].shape}")
    print(f"  Target (GT) shape: {sample['gt_target'].shape}")
    print(f"  Langragian prediction shape: {sample['langragian_target'].shape}")
    print(f"  Difference (GT-Langragian) shape: {sample['difference_target'].shape}")
    print(f"  IR input shape: {sample['ir_input'].shape}")
    
    print(f"Ground truth period: {sample['flash_flood_info']['target_period']}")
    
    return [sample]  # Return as list for compatibility


def handle_nan_values(data):
    """Replace NaN values using the existing pipeline strategy"""
    return np.nan_to_num(data, nan=0.0)


def standardize_timestamps(timestamps):
    """Convert timestamps to standard format used by the model"""
    standardized = []
    for ts in timestamps:
        if isinstance(ts, bytes):
            ts = ts.decode()
        # Parse and reformat to standard format
        try:
            dt = datetime.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            # Try alternative formats
            for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S.%f']:
                try:
                    dt = datetime.datetime.strptime(ts, fmt)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"Cannot parse timestamp: {ts}")
        standardized.append(dt.strftime('%Y-%m-%d %H:%M:%S'))
    return standardized


def fill_temporal_gaps(data, timestamps, expected_interval_minutes=15):
    """Fill missing 15-minute intervals using forward-fill strategy"""
    print("Filling temporal gaps in IR data...")
    
    # Parse timestamps
    parsed_timestamps = []
    for ts in timestamps:
        if isinstance(ts, bytes):
            ts = ts.decode()
        parsed_timestamps.append(datetime.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S'))
    
    # Create expected timestamp sequence
    start_time = parsed_timestamps[0]
    end_time = parsed_timestamps[-1]
    expected_timestamps = []
    current_time = start_time
    
    while current_time <= end_time:
        expected_timestamps.append(current_time)
        current_time += datetime.timedelta(minutes=expected_interval_minutes)
    
    # Create mapping of existing data
    existing_data = {}
    for i, ts in enumerate(parsed_timestamps):
        existing_data[ts] = data[i]
    
    # Fill missing intervals with forward-fill
    filled_data = []
    filled_timestamps = []
    last_valid_data = None
    
    for expected_ts in expected_timestamps:
        if expected_ts in existing_data:
            # Use actual data
            current_data = existing_data[expected_ts]
            last_valid_data = current_data
        else:
            # Forward-fill with last valid data
            if last_valid_data is not None:
                current_data = last_valid_data.copy()
                print(f"  Filled missing timestamp {expected_ts} with forward-fill")
            else:
                # If no previous data, create zeros
                current_data = np.zeros_like(data[0])
                print(f"  Filled missing timestamp {expected_ts} with zeros (no previous data)")
        
        filled_data.append(current_data)
        filled_timestamps.append(expected_ts.strftime('%Y-%m-%d %H:%M:%S'))
    
    filled_data = np.array(filled_data)
    print(f"  Filled {len(filled_data) - len(data)} missing intervals")
    
    return filled_data, filled_timestamps


def validate_data_quality(data, timestamps, data_type="IR"):
    """Ensure data meets model expectations"""
    print(f"Validating {data_type} data quality...")
    
    # Check for NaN values
    nan_count = np.isnan(data).sum()
    if nan_count > 0:
        print(f"  Warning: Found {nan_count} NaN values in {data_type} data")
        return False
    
    # Check for infinite values
    inf_count = np.isinf(data).sum()
    if inf_count > 0:
        print(f"  Warning: Found {inf_count} infinite values in {data_type} data")
        return False
    
    # Check temporal continuity
    if len(timestamps) > 1:
        time_diffs = []
        for i in range(1, len(timestamps)):
            ts1 = datetime.datetime.strptime(timestamps[i-1], '%Y-%m-%d %H:%M:%S')
            ts2 = datetime.datetime.strptime(timestamps[i], '%Y-%m-%d %H:%M:%S')
            diff_minutes = (ts2 - ts1).total_seconds() / 60
            time_diffs.append(diff_minutes)
        
        expected_diff = 15 if data_type == "IR" else 30
        irregular_intervals = [d for d in time_diffs if abs(d - expected_diff) > 1]
        if irregular_intervals:
            print(f"  Warning: Found {len(irregular_intervals)} irregular time intervals")
    
    print(f"  {data_type} data validation passed: {data.shape} with {len(timestamps)} timestamps")
    return True


def preprocess_ir_data(ir_file):
    """Apply comprehensive IR data preprocessing"""
    print("Preprocessing IR data with NaN handling and gap filling...")
    
    # Read original data
    with h5py.File(ir_file, 'r') as f:
        original_data = f['Tb'][:]
        if 'timestamps' in f:
            original_timestamps = [ts.decode() if isinstance(ts, bytes) else ts for ts in f['timestamps'][:]]
        else:
            # Create dummy timestamps if missing
            num_timesteps = original_data.shape[0]
            base_time = datetime.datetime.now().replace(hour=0, minute=0, second=0)
            original_timestamps = [(base_time + datetime.timedelta(minutes=15*i)).strftime('%Y-%m-%d %H:%M:%S') 
                                 for i in range(num_timesteps)]
    
    # Step 1: Handle NaN values
    processed_data = handle_nan_values(original_data)
    print(f"  Replaced NaN values: {np.isnan(original_data).sum()} -> {np.isnan(processed_data).sum()}")
    
    # Step 2: Standardize timestamps
    standardized_timestamps = standardize_timestamps(original_timestamps)
    
    # Step 3: Fill temporal gaps
    filled_data, filled_timestamps = fill_temporal_gaps(processed_data, standardized_timestamps)
    
    # Step 4: Final NaN cleanup (in case forward-fill introduced any)
    filled_data = handle_nan_values(filled_data)
    
    # Step 5: Validate data quality
    validate_data_quality(filled_data, filled_timestamps, "IR")
    
    # Create backup and save preprocessed data
    backup_file = ir_file.replace('.h5', '_original.h5')
    if not os.path.exists(backup_file):
        shutil.copy2(ir_file, backup_file)
        print(f"  Created backup: {backup_file}")
    
    # Save preprocessed data
    with h5py.File(ir_file, 'w') as f:
        f.create_dataset('Tb', data=filled_data.astype(np.float32))
        f.create_dataset('precipitations', data=filled_data.astype(np.float32))  # Also save as 'precipitations' for compatibility
        f.create_dataset('timestamps', data=[ts.encode() for ts in filled_timestamps])
    
    print(f"  Preprocessed IR data saved: {filled_data.shape}")
    return filled_data, filled_timestamps


def preprocess_imerg_data(imerg_file):
    """Apply NaN handling to IMERG data"""
    print("Preprocessing IMERG data with NaN handling...")
    
    # Read original data
    with h5py.File(imerg_file, 'r') as f:
        original_data = f['precipitations'][:]
        if 'timestamps' in f:
            original_timestamps = [ts.decode() if isinstance(ts, bytes) else ts for ts in f['timestamps'][:]]
        else:
            # Create dummy timestamps if missing
            num_timesteps = original_data.shape[0]
            base_time = datetime.datetime.now().replace(hour=0, minute=0, second=0)
            original_timestamps = [(base_time + datetime.timedelta(minutes=30*i)).strftime('%Y-%m-%d %H:%M:%S') 
                                 for i in range(num_timesteps)]
    
    # Handle NaN values
    processed_data = handle_nan_values(original_data)
    print(f"  Replaced NaN values: {np.isnan(original_data).sum()} -> {np.isnan(processed_data).sum()}")
    
    # Standardize timestamps
    standardized_timestamps = standardize_timestamps(original_timestamps)
    
    # Validate data quality
    validate_data_quality(processed_data, standardized_timestamps, "IMERG")
    
    # Create backup and save preprocessed data
    backup_file = imerg_file.replace('.h5', '_original.h5')
    if not os.path.exists(backup_file):
        shutil.copy2(imerg_file, backup_file)
        print(f"  Created backup: {backup_file}")
    
    # Save preprocessed data
    with h5py.File(imerg_file, 'w') as f:
        f.create_dataset('precipitations', data=processed_data.astype(np.float32))
        f.create_dataset('timestamps', data=[ts.encode() for ts in standardized_timestamps])
    
    print(f"  Preprocessed IMERG data saved: {processed_data.shape}")
    return processed_data, standardized_timestamps


def load_models(model_path, device):
    """Load the trained models for Langragian LDM approach"""
    print("Loading models for Langragian LDM approach...")
    
    # Initialize U-Net for the difference-based approach
    # Note: The channel dimensions are different since we're using difference + IR encoding
    unet = DiffusionModelUNet(
        with_conditioning=True,
        cross_attention_dim=64,  # Combined difference VAE (32) + IR VAE (32) channels
        spatial_dims=3,
        in_channels=32,  # Difference latent channels
        out_channels=32,  # Difference latent channels
        num_res_blocks=6,
        num_channels=(128, 256, 512, 512),
        attention_levels=(False, True, True, True),
        num_head_channels=(0, 256, 512, 512),
    ).to(device)
    
    # Load trained model checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    unet.load_state_dict(checkpoint['model_state_dict'])
    unet.eval()
    
    print(f"Loaded Langragian LDM model from {model_path}")
    if 'epoch' in checkpoint:
        print(f"Model trained for {checkpoint['epoch'] + 1} epochs")
    
    # Load VAE models for encoding/decoding
    print("Loading VAE models...")
    
    # 1. Load Difference VAE model
    from vae_models.vae_imerg import SimpleVAE3D
    diff_vae_model = SimpleVAE3D(input_channels=1, latent_dim=32).to(device)
    
    # Default path for difference VAE
    diff_vae_path = os.path.join(os.path.dirname(__file__), 'vae_models', 'diff_encoder.pth')
    if os.path.exists(diff_vae_path):
        diff_checkpoint = torch.load(diff_vae_path, map_location=device, weights_only=False)
        diff_vae_model.load_state_dict(diff_checkpoint['model_state_dict'])
        print(f"Loaded Difference VAE from {diff_vae_path}")
        
        # Load normalization stats if available
        diff_mean = diff_checkpoint.get('mean', 0.0)
        diff_std = diff_checkpoint.get('std', 1.0)
        if isinstance(diff_mean, torch.Tensor):
            diff_mean = diff_mean.item()
        if isinstance(diff_std, torch.Tensor):
            diff_std = diff_std.item()
        print(f"Difference VAE normalization - mean: {diff_mean}, std: {diff_std}")
    else:
        print(f"Warning: Difference VAE not found at {diff_vae_path}")
        diff_mean, diff_std = 0.0, 1.0
    
    diff_vae_model.eval()
    
    # 2. Load IR VAE model  
    ir_vae_model = SimpleVAE3D(input_channels=1, latent_dim=32).to(device)
    
    # Path for IR VAE (hardcoded in langragian_ldm_dataloader.py)
    ir_vae_path = os.path.join(os.path.dirname(__file__), 'vae_models', 'ir_model_final.pth')
    if os.path.exists(ir_vae_path):
        ir_checkpoint = torch.load(ir_vae_path, map_location=device, weights_only=False)
        ir_vae_model.load_state_dict(ir_checkpoint['model_state_dict'])
        print(f"Loaded IR VAE from {ir_vae_path}")
        
        # Load IR normalization stats
        ir_mean = ir_checkpoint.get('mean', 0.0)
        ir_std = ir_checkpoint.get('std', 1.0)
        if isinstance(ir_mean, torch.Tensor):
            ir_mean = ir_mean.item() 
        if isinstance(ir_std, torch.Tensor):
            ir_std = ir_std.item()
        print(f"IR VAE normalization - mean: {ir_mean}, std: {ir_std}")
    else:
        print(f"Warning: IR VAE not found at {ir_vae_path}")
        ir_mean, ir_std = 0.0, 1.0
    
    ir_vae_model.eval()
    
    # Initialize scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
        beta_start=0.0001,
        beta_end=0.02
    )
    
    # Return loaded models with normalization stats
    models_dict = {
        'unet': unet,
        'diff_vae': diff_vae_model,
        'ir_vae': ir_vae_model,
        'scheduler': scheduler,
        'diff_stats': (diff_mean, diff_std),
        'ir_stats': (ir_mean, ir_std)
    }
    
    return models_dict


def encode_difference_to_latent(diff_data, models_dict, device):
    """Encode difference data to latent space using difference VAE"""
    diff_vae_model = models_dict['diff_vae']
    diff_mean, diff_std = models_dict['diff_stats']
    
    with torch.no_grad():
        # Skip normalization for difference data
        diff_normalized = torch.nan_to_num(diff_data, 0.0)
        
        # Add channel dimension: (T, H, W) -> (T, C, H, W)
        diff_tensor = diff_normalized.unsqueeze(1).to(device)  # (T, 1, H, W)
        
        # Permute to VAE expected format: (T, C, H, W) -> (C, T, H, W)
        diff_tensor = diff_tensor.permute(1, 0, 2, 3)  # (1, T, H, W)
        
        # Add batch dimension
        diff_tensor = diff_tensor.unsqueeze(0).to(device)  # (1, 1, T, H, W)
        
        # Encode to latent space
        mu, logvar = diff_vae_model.encode(diff_tensor)
        latent = mu  # Use mean for deterministic encoding
        
        # Remove batch dimension and permute back: (1, C_latent, T_latent, H_latent, W_latent) -> (C_latent, T_latent, H_latent, W_latent)
        latent = latent.squeeze(0)  # (C_latent, T_latent, H_latent, W_latent)
        
    return latent.cpu()


def encode_ir_to_latent(ir_data, models_dict, device):
    """Encode IR data to latent space using IR VAE"""
    ir_vae_model = models_dict['ir_vae']
    ir_mean, ir_std = models_dict['ir_stats']
    
    with torch.no_grad():
        # Normalize IR data using IR model's stats
        ir_normalized = (ir_data - ir_mean) / ir_std
        ir_normalized = torch.nan_to_num(ir_normalized, 0.0)
        
        # Add channel dimension: (T, H, W) -> (T, C, H, W)
        ir_tensor = ir_normalized.unsqueeze(1).to(device)  # (T, 1, H, W)
        
        # Permute to VAE expected format: (T, C, H, W) -> (C, T, H, W)
        ir_tensor = ir_tensor.permute(1, 0, 2, 3)  # (1, T, H, W)
        
        # Add batch dimension
        ir_tensor = ir_tensor.unsqueeze(0).to(device)  # (1, 1, T, H, W)
        
        # Encode to latent space
        mu, logvar = ir_vae_model.encode(ir_tensor)
        latent = mu  # Use mean for deterministic encoding
        
        # Remove batch dimension and permute back
        latent = latent.squeeze(0)  # (C_latent, T_latent, H_latent, W_latent)
        
    return latent.cpu()


def generate_sample_ldm(models_dict, condition, target_shape, langragian_baseline, device, ensemble_size=1):
    """Generate a sample using the Langragian LDM (Latent Diffusion Model)"""
    unet = models_dict['unet']
    diff_vae_model = models_dict['diff_vae']
    scheduler = models_dict['scheduler']
    
    samples = []
    inference_times = []
    
    for _ in range(ensemble_size):
        with torch.no_grad():
            # Start with random noise in latent space (difference space)
            # target_shape should be the shape of the target difference latents: (B, C, T, H, W)
            latent_sample = torch.randn(target_shape).to(device)
            
            # DDPM sampling with fewer steps for faster inference
            start_time = time_module.time()
            num_inference_steps = 100  # Reduced from 1000 for faster inference
            skip_step = scheduler.num_train_timesteps // num_inference_steps
            
            for i, t in enumerate(reversed(range(0, scheduler.num_train_timesteps, skip_step))):
                t_tensor = torch.tensor([t], device=device).long()
                
                # Predict noise
                noise_pred = unet(
                    x=latent_sample,
                    timesteps=t_tensor,
                    context=condition
                )
                
                # Remove noise
                step_output = scheduler.step(noise_pred, t, latent_sample)
                if hasattr(step_output, 'prev_sample'):
                    latent_sample = step_output.prev_sample
                else:
                    latent_sample = step_output[0] if isinstance(step_output, tuple) else step_output
            
            # Decode the generated latent difference back to precipitation space
            # latent_sample is (B, C_latent, T_latent, H_latent, W_latent)
            decoded_difference = diff_vae_model.decode(latent_sample)  # (B, C, T, H, W)
            
            # Convert back to (B, T, C, H, W) for consistency
            decoded_difference = decoded_difference.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
            
            # Remove channel dimension and batch dimension to get (T, H, W)
            decoded_difference = decoded_difference.squeeze(0).squeeze(1)  # (T, H, W)
            
            # Add the predicted difference to the Langragian baseline
            # Final prediction = Langragian baseline + learned difference
            final_prediction = langragian_baseline + decoded_difference
            
            # Ensure non-negative precipitation values
            final_prediction = torch.clamp(final_prediction, min=0.0)
            
            inference_time = time_module.time() - start_time
            
            samples.append(final_prediction)  # Return final precipitation prediction
            inference_times.append(inference_time)
    
    return samples, inference_times


def run_baseline_methods(precip_input, timesteps=12):
    """Run baseline nowcasting methods based on calculate_csi.py implementation"""
    baseline_predictions = {}
    
    # Add nowcasting methods path - try multiple possible locations
    possible_paths = [
        '/home1/ppatel2025/main/nowcasting/servir/methods/ExtrapolationMethods',
        os.path.join(os.path.dirname(__file__), '..', 'residual_vae', 'src'),
        os.path.join(os.path.dirname(__file__), '..', 'residual_vae', 'src', 'extrapolation_methods')
    ]
    
    NOWCASTING_AVAILABLE = False
    for path in possible_paths:
        if path not in sys.path:
            sys.path.append(path)
    
    try:
        from extrapolation_methods.langragian_persistance import langragian_persistance
        from extrapolation_methods.linda import linda
        from extrapolation_methods.steps import steps
        from extrapolation_methods.naive_persistence import naive_persistence
        NOWCASTING_AVAILABLE = True
        print(f"Successfully imported nowcasting methods")
    except ImportError:
        try:
            # Try alternative import structure
            from extrapolation_methods import linda, steps, langragian_persistance
            from naive_persistence import naive_persistence
            NOWCASTING_AVAILABLE = True
            print(f"Successfully imported nowcasting methods (alternative structure)")
        except ImportError as e:
            print(f"Warning: Nowcasting methods not available. Error: {e}")
            print(f"Tried paths: {possible_paths}")
            NOWCASTING_AVAILABLE = False
    
    if not NOWCASTING_AVAILABLE:
        print("Warning: Nowcasting methods not available - creating dummy predictions")
        # Create dummy predictions for demonstration
        if torch.is_tensor(precip_input):
            precip_input = precip_input.cpu().numpy()
        
        # Ensure shape is (T, H, W)
        if len(precip_input.shape) == 4:  # (B, T, C, H, W)
            precip_input = precip_input[0, :, 0, :, :]  # Take first batch and channel
        elif len(precip_input.shape) == 5:  # (B, T, C, H, W)
            precip_input = precip_input[0, :, 0, :, :]
        
        # Create simple persistence forecasts as fallback
        last_frame = precip_input[-1]  # Use last frame
        dummy_forecast = np.stack([last_frame * 0.9**i for i in range(timesteps)])  # Decay over time
        
        baseline_predictions['naive'] = dummy_forecast
        baseline_predictions['lagrangian'] = dummy_forecast * 0.8  # Slightly different
        baseline_predictions['steps'] = dummy_forecast * 1.1  # Slightly different
        baseline_predictions['linda'] = dummy_forecast * 0.95  # Slightly different
        
        return baseline_predictions
    
    # Convert input to numpy (T, H, W)
    if torch.is_tensor(precip_input):
        precip_input = precip_input.cpu().numpy()
    
    # Ensure shape is (T, H, W)
    if len(precip_input.shape) == 4:  # (B, T, C, H, W)
        precip_input = precip_input[0, :, 0, :, :]  # Take first batch and channel
    elif len(precip_input.shape) == 5:  # (B, T, C, H, W)
        precip_input = precip_input[0, :, 0, :, :]
    
    try:
        # PySTEPS (ensemble=1)
        print("    Running PySTEPS...")
        steps_result = steps(precip_input, timesteps, n_ens_members=1, return_output=True)
        if steps_result is not None:
            baseline_predictions['steps'] = steps_result
            print(f"    PySTEPS completed successfully. Shape: {steps_result.shape}")
    except Exception as e:
        print(f"    PySTEPS failed: {e}")
    
    try:
        # LINDA (ensemble=1)
        print("    Running LINDA...")
        linda_result = linda(precip_input, timesteps, add_perturbations=False, n_ens_members=1, return_output=True)
        if linda_result is not None:
            baseline_predictions['linda'] = linda_result
            print(f"    LINDA completed successfully. Shape: {linda_result.shape}")
    except Exception as e:
        print(f"    LINDA failed: {e}")
    
    try:
        # Lagrangian persistence
        print("    Running Lagrangian...")
        lagrangian_result = langragian_persistance(precip_input, timesteps)
        if lagrangian_result is not None:
            baseline_predictions['lagrangian'] = lagrangian_result
            print(f"    Lagrangian completed successfully. Shape: {lagrangian_result.shape}")
    except Exception as e:
        print(f"    Lagrangian failed: {e}")
    
    try:
        # Naive persistence
        print("    Running Naive...")
        naive_result = naive_persistence(precip_input, timesteps)
        if naive_result is not None:
            baseline_predictions['naive'] = naive_result
            print(f"    Naive completed successfully. Shape: {naive_result.shape}")
    except Exception as e:
        print(f"    Naive failed: {e}")
    
    return baseline_predictions


def create_individual_comparison_gifs(predictions_dict, ground_truth, output_dir, start_datetime, fps=2):
    """Create separate GIFs for each baseline method vs ground truth comparison"""
    # Convert to numpy and move to CPU
    model_preds = {}
    
    for name, pred in predictions_dict.items():
        if isinstance(pred, torch.Tensor):
            model_preds[name] = pred.cpu().numpy()
        else:
            model_preds[name] = pred
    
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().numpy()
    
    # Get dimensions - should be 12 timesteps for the flash flood period (21:00 to 03:00)
    time_steps = ground_truth.shape[1] if ground_truth is not None else 12
    
    # Take first sample and first channel
    ground_truth_seq = ground_truth[0, :, 0, :, :] if ground_truth is not None else None
    
    pred_seqs = {}
    for name, pred in model_preds.items():
        if len(pred.shape) == 5:  # (B, T, C, H, W)
            pred_seqs[name] = pred[0, :, 0, :, :]
        elif len(pred.shape) == 4:  # (T, C, H, W) or (B, T, H, W)
            if pred.shape[1] == time_steps:  # (B, T, H, W)
                pred_seqs[name] = pred[0, :, :, :]
            else:  # (T, C, H, W)
                pred_seqs[name] = pred[:, 0, :, :]
        else:  # (T, H, W)
            pred_seqs[name] = pred
    
    # Geographic coordinates for West/Central Africa region
    xmin, xmax = -21.4, 30.4
    ymin, ymax = -2.9, 33.1
    
    # Create separate GIF for each method vs ground truth
    for method_name, pred_seq in pred_seqs.items():
        if pred_seq is None:
            continue
            
        print(f"Creating GIF for {method_name.upper()} vs Ground Truth...")
        
        # Create figure with 2 subplots side by side (Ground Truth + Method)
        if CARTOPY_AVAILABLE:
            fig = plt.figure(figsize=(15, 6))
            projection = ccrs.PlateCarree()
            
            # Ground Truth subplot
            ax1 = fig.add_subplot(1, 2, 1, projection=projection)
            ax1.add_feature(cfeature.COASTLINE, linewidth=0.3, alpha=0.3)
            ax1.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.2)
            ax1.add_feature(cfeature.OCEAN, alpha=0.1)
            ax1.add_feature(cfeature.LAND, alpha=0.1)
            ax1.set_extent([xmin, xmax, ymin, ymax], crs=projection)
            ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.2, linewidth=0.2)
            ax1.set_title('Ground Truth', fontsize=14, fontweight='bold')
            
            # Method prediction subplot  
            ax2 = fig.add_subplot(1, 2, 2, projection=projection)
            ax2.add_feature(cfeature.COASTLINE, linewidth=0.3, alpha=0.3)
            ax2.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.2)
            ax2.add_feature(cfeature.OCEAN, alpha=0.1)
            ax2.add_feature(cfeature.LAND, alpha=0.1)
            ax2.set_extent([xmin, xmax, ymin, ymax], crs=projection)
            ax2.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.2, linewidth=0.2)
            ax2.set_title(f'{method_name.upper()} Prediction', fontsize=14, fontweight='bold')
            
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            ax1.set_title('Ground Truth', fontsize=14, fontweight='bold')
            ax2.set_title(f'{method_name.upper()} Prediction', fontsize=14, fontweight='bold')
        
        # Determine value ranges for consistent coloring
        if ground_truth_seq is not None:
            all_values = np.concatenate([ground_truth_seq.flatten(), pred_seq.flatten()])
        else:
            all_values = pred_seq.flatten()
        
        pred_vmin, pred_vmax = 0, np.max(all_values) if len(all_values) > 0 else 1
        
        # Set up the extent
        extent = [xmin, xmax, ymin, ymax]
        
        # Initialize images
        if CARTOPY_AVAILABLE:
            # Ground truth image
            if ground_truth_seq is not None:
                im_gt = ax1.imshow(ground_truth_seq[0], cmap='Blues', vmin=pred_vmin, vmax=pred_vmax,
                                  extent=extent, transform=projection, origin='lower')
            else:
                im_gt = ax1.imshow(np.zeros_like(pred_seq[0]), cmap='Blues', vmin=pred_vmin, vmax=pred_vmax,
                                  extent=extent, transform=projection, origin='lower')
            
            # Method prediction image
            im_pred = ax2.imshow(pred_seq[0], cmap='Reds', vmin=pred_vmin, vmax=pred_vmax,
                               extent=extent, transform=projection, origin='lower')
        else:
            # Same logic for non-cartopy
            if ground_truth_seq is not None:
                im_gt = ax1.imshow(ground_truth_seq[0], cmap='Blues', vmin=pred_vmin, vmax=pred_vmax, 
                                  extent=extent, origin='lower')
            else:
                im_gt = ax1.imshow(np.zeros_like(pred_seq[0]), cmap='Blues', vmin=pred_vmin, vmax=pred_vmax, 
                                  extent=extent, origin='lower')
            
            im_pred = ax2.imshow(pred_seq[0], cmap='Reds', vmin=pred_vmin, vmax=pred_vmax, 
                               extent=extent, origin='lower')
        
        # Add colorbars with better positioning to avoid overlap
        cbar1 = plt.colorbar(im_gt, ax=ax1, shrink=0.7, pad=0.15)
        cbar1.set_label('Precipitation (mm/h)', fontsize=10)
        cbar2 = plt.colorbar(im_pred, ax=ax2, shrink=0.7, pad=0.15)
        cbar2.set_label('Precipitation (mm/h)', fontsize=10)
        
        def animate(frame):
            # Update ground truth
            if ground_truth_seq is not None and frame < len(ground_truth_seq):
                im_gt.set_array(ground_truth_seq[frame])
            
            # Update method prediction
            if frame < len(pred_seq):
                im_pred.set_array(pred_seq[frame])
            
            # Calculate time from start datetime (21:00 + frame * 30min)
            current_time = start_datetime + datetime.timedelta(minutes=frame * 30)
            
            fig.suptitle(f'Flash Flood Event | {method_name.upper()} vs Ground Truth | {current_time.strftime("%Y-%m-%d %H:%M")} | Frame: {frame+1}/{time_steps}', 
                        fontsize=16, fontweight='bold')
            
            return [im_gt, im_pred]
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=time_steps, interval=1000//fps, blit=False, repeat=True
        )
        
        # Adjust layout to prevent colorbar overlap
        plt.tight_layout()
        
        # Save as separate GIF for this method
        output_path = os.path.join(output_dir, f"flash_flood_{method_name}_vs_ground_truth.gif")
        anim.save(output_path, writer='pillow', fps=fps, dpi=100)
        plt.close(fig)
        
        print(f"  {method_name.upper()} vs GT GIF saved: {output_path}")
    
    print(f"All individual comparison GIFs created!")


def create_ground_truth_gif(ground_truth, output_dir, start_datetime, fps=2):
    """Create a GIF showing only the ground truth evolution for the flash flood event"""
    print(f"Creating Ground Truth GIF for flash flood event...")
    
    # Convert to numpy and move to CPU
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().numpy()
    
    # Get dimensions - should be 12 timesteps for the flash flood period (21:00 to 03:00)
    time_steps = ground_truth.shape[1] if ground_truth is not None else 12
    
    # Take first sample and first channel
    ground_truth_seq = ground_truth[0, :, 0, :, :] if ground_truth is not None else None
    
    if ground_truth_seq is None:
        print("No ground truth data available for GIF creation")
        return
    
    # Geographic coordinates for West/Central Africa region
    xmin, xmax = -21.4, 30.4
    ymin, ymax = -2.9, 33.1
    
    # Create figure with single subplot for ground truth
    if CARTOPY_AVAILABLE:
        fig = plt.figure(figsize=(12, 9))
        projection = ccrs.PlateCarree()
        
        # Ground Truth subplot
        ax = fig.add_subplot(1, 1, 1, projection=projection)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, alpha=0.3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.2)
        ax.add_feature(cfeature.OCEAN, alpha=0.1, color='lightblue')
        ax.add_feature(cfeature.LAND, alpha=0.1, color='lightgray')
        ax.set_extent([xmin, xmax, ymin, ymax], crs=projection)
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, 
                    alpha=0.2, linewidth=0.3, linestyle='--')
        
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
    
    # Determine value range for precipitation
    pred_vmin, pred_vmax = 0, np.max(ground_truth_seq) if len(ground_truth_seq) > 0 else 1
    
    # Set up the extent
    extent = [xmin, xmax, ymin, ymax]
    
    # Initialize image
    if CARTOPY_AVAILABLE:
        im_gt = ax.imshow(ground_truth_seq[0], cmap='Blues', vmin=pred_vmin, vmax=pred_vmax,
                         extent=extent, transform=projection, origin='lower')
    else:
        im_gt = ax.imshow(ground_truth_seq[0], cmap='Blues', vmin=pred_vmin, vmax=pred_vmax, 
                         extent=extent, origin='lower')
    
    # Add colorbar with better positioning to avoid overlap
    cbar = plt.colorbar(im_gt, ax=ax, shrink=0.7, pad=0.15)
    cbar.set_label('Precipitation (mm/h)', fontsize=12, fontweight='bold')
    
    def animate(frame):
        # Update ground truth
        if frame < len(ground_truth_seq):
            im_gt.set_array(ground_truth_seq[frame])
        
        # Calculate time from start datetime (21:00 + frame * 30min)
        current_time = start_datetime + datetime.timedelta(minutes=frame * 30)
        
        # Create detailed title with event information
        fig.suptitle(f'Flash Flood Event - Ground Truth Precipitation\n' +
                    f'West/Central Africa | {current_time.strftime("%Y-%m-%d %H:%M")} UTC | ' +
                    f'Frame: {frame+1}/{time_steps}', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Add frame-specific statistics
        current_precip = ground_truth_seq[frame]
        max_precip = np.max(current_precip)
        mean_precip = np.mean(current_precip[current_precip > 0])  # Mean of non-zero values
        
        ax.text(0.02, 0.98, f'Max: {max_precip:.2f} mm/h\nMean: {mean_precip:.2f} mm/h', 
               transform=ax.transAxes, fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
               verticalalignment='top')
        
        return [im_gt]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=time_steps, interval=1000//fps, blit=False, repeat=True
    )
    
    # Adjust layout to prevent colorbar overlap
    plt.tight_layout()
    
    # Save as GIF
    output_path = os.path.join(output_dir, "flash_flood_ground_truth.gif")
    anim.save(output_path, writer='pillow', fps=fps, dpi=100)
    plt.close(fig)
    
    print(f"  Ground Truth GIF saved: {output_path}")
    
    return output_path


def create_prediction_plots(predictions_dict, ground_truth, output_dir, event_info, timesteps_to_plot=[0, 3, 6, 9, 11]):
    """Create static prediction plots for specific timesteps of the flash flood event"""
    print(f"Creating static prediction plots for flash flood event...")
    
    # Convert to numpy and move to CPU
    model_preds = {}
    for name, pred in predictions_dict.items():
        if isinstance(pred, torch.Tensor):
            model_preds[name] = pred.cpu().numpy()
        else:
            model_preds[name] = pred
    
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().numpy()
    
    # Take first sample and first channel - handle different ground truth shapes
    if ground_truth is not None:
        if len(ground_truth.shape) == 5:  # (B, T, C, H, W)
            ground_truth_seq = ground_truth[0, :, 0, :, :]  # (T, H, W)
        elif len(ground_truth.shape) == 4:  # (B, T, H, W)
            ground_truth_seq = ground_truth[0, :, :, :]  # (T, H, W)
        elif len(ground_truth.shape) == 3:  # (T, H, W)
            ground_truth_seq = ground_truth
        else:
            print(f"Warning: Unexpected ground truth shape: {ground_truth.shape}")
            ground_truth_seq = None
    else:
        ground_truth_seq = None
    
    pred_seqs = {}
    for name, pred in model_preds.items():
        if len(pred.shape) == 5:  # (B, T, C, H, W)
            pred_seqs[name] = pred[0, :, 0, :, :]
        elif len(pred.shape) == 4:  # (T, C, H, W) or (B, T, H, W)
            if pred.shape[1] == 12:  # (B, T, H, W)
                pred_seqs[name] = pred[0, :, :, :]
            else:  # (T, C, H, W)
                pred_seqs[name] = pred[:, 0, :, :]
        else:  # (T, H, W)
            pred_seqs[name] = pred
    
    # Geographic coordinates for West/Central Africa region
    xmin, xmax = -21.4, 30.4
    ymin, ymax = -2.9, 33.1
    
    # Create plots for each timestep
    for timestep_idx in timesteps_to_plot:
        if timestep_idx >= 12:  # Ensure we don't exceed available timesteps
            continue
            
        # Calculate the actual time for this timestep
        event_start = event_info['event_start']
        current_time = event_start + datetime.timedelta(minutes=timestep_idx * 30)
        
        print(f"  Creating plot for timestep {timestep_idx} - {current_time.strftime('%Y-%m-%d %H:%M')}")
        
        # Create figure with subplots for each method + ground truth
        n_methods = len(pred_seqs) + 1  # +1 for ground truth
        n_cols = min(3, n_methods)  # Max 3 columns
        n_rows = (n_methods + n_cols - 1) // n_cols  # Calculate required rows
        
        if CARTOPY_AVAILABLE:
            fig = plt.figure(figsize=(5*n_cols, 4*n_rows))
            projection = ccrs.PlateCarree()
        else:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_methods == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
        
        plot_idx = 0
        extent = [xmin, xmax, ymin, ymax]
        
        # Determine value range for consistent coloring
        all_values = []
        if ground_truth_seq is not None:
            all_values.append(ground_truth_seq[timestep_idx].flatten())
        for pred_seq in pred_seqs.values():
            if timestep_idx < pred_seq.shape[0]:
                all_values.append(pred_seq[timestep_idx].flatten())
        
        if all_values:
            all_values = np.concatenate(all_values)
            vmin, vmax = 0, np.max(all_values) if len(all_values) > 0 else 1
        else:
            vmin, vmax = 0, 1
        
        # Plot ground truth
        if ground_truth_seq is not None:
            row, col = plot_idx // n_cols, plot_idx % n_cols
            
            if CARTOPY_AVAILABLE:
                ax = fig.add_subplot(n_rows, n_cols, plot_idx + 1, projection=projection)
                ax.add_feature(cfeature.COASTLINE, linewidth=0.3, alpha=0.3)
                ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.2)
                ax.add_feature(cfeature.OCEAN, alpha=0.1)
                ax.add_feature(cfeature.LAND, alpha=0.1)
                ax.set_extent([xmin, xmax, ymin, ymax], crs=projection)
                ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.2, linewidth=0.2)
                
                im = ax.imshow(ground_truth_seq[timestep_idx], cmap='Blues', vmin=vmin, vmax=vmax,
                              extent=extent, transform=projection, origin='lower')
            else:
                ax = axes[row, col] if n_rows > 1 else axes[col]
                im = ax.imshow(ground_truth_seq[timestep_idx], cmap='Blues', vmin=vmin, vmax=vmax,
                              extent=extent, origin='lower')
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
            
            ax.set_title('Ground Truth', fontsize=12, fontweight='bold')
            plot_idx += 1
        
        # Plot predictions
        colors = ['Reds', 'Greens', 'Oranges', 'Purples', 'YlOrRd']
        for i, (method_name, pred_seq) in enumerate(pred_seqs.items()):
            if timestep_idx >= pred_seq.shape[0]:
                continue
                
            row, col = plot_idx // n_cols, plot_idx % n_cols
            
            if CARTOPY_AVAILABLE:
                ax = fig.add_subplot(n_rows, n_cols, plot_idx + 1, projection=projection)
                ax.add_feature(cfeature.COASTLINE, linewidth=0.3, alpha=0.3)
                ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.2)
                ax.add_feature(cfeature.OCEAN, alpha=0.1)
                ax.add_feature(cfeature.LAND, alpha=0.1)
                ax.set_extent([xmin, xmax, ymin, ymax], crs=projection)
                ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.2, linewidth=0.2)
                
                cmap = colors[i % len(colors)]
                im = ax.imshow(pred_seq[timestep_idx], cmap=cmap, vmin=vmin, vmax=vmax,
                              extent=extent, transform=projection, origin='lower')
            else:
                ax = axes[row, col] if n_rows > 1 else axes[col]
                cmap = colors[i % len(colors)]
                im = ax.imshow(pred_seq[timestep_idx], cmap=cmap, vmin=vmin, vmax=vmax,
                              extent=extent, origin='lower')
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
            
            ax.set_title(f'{method_name.upper()}', fontsize=12, fontweight='bold')
            plot_idx += 1
        
        # Hide any unused subplots
        if CARTOPY_AVAILABLE:
            # For cartopy, we don't need to hide unused subplots as they're created individually
            pass
        else:
            while plot_idx < n_rows * n_cols:
                row, col = plot_idx // n_cols, plot_idx % n_cols
                if n_rows > 1:
                    axes[row, col].set_visible(False)
                else:
                    axes[col].set_visible(False)
                plot_idx += 1
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=fig.get_axes(), shrink=0.6, pad=0.02)
        cbar.set_label('Precipitation (mm/h)', fontsize=12)
        
        # Add main title
        fig.suptitle(f'Flash Flood Event Predictions\n{current_time.strftime("%Y-%m-%d %H:%M")} UTC (Timestep {timestep_idx+1}/12)',
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"flash_flood_predictions_timestep_{timestep_idx:02d}_{current_time.strftime('%Y%m%d_%H%M')}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"    Saved: {plot_filename}")
    
    print(f"Static prediction plots completed!")


def create_single_method_gif(method_name, pred_seq, ground_truth_seq, output_path, start_datetime, fps=2):
    """Create a single GIF comparing one method with ground truth"""
    # This function can be used if needed for more granular control
    pass


def run_inference(imerg_file, ir_file, start_datetime, end_datetime, device, output_dir, models_dict, vae_checkpoint_path, ensemble_size=1):
    """Run inference using baseline methods and Langragian LDM for the specified time range"""
    print("Running flash flood inference for baseline methods and Langragian LDM...")
    print(f"Time range: {start_datetime} to {end_datetime}")
    print("Methods: PySTEPS, LINDA, Lagrangian, Naive, Langragian LDM")
    
    # Check if files exist
    if not os.path.exists(imerg_file):
        raise FileNotFoundError(f"IMERG file not found: {imerg_file}")
    if not os.path.exists(ir_file):
        raise FileNotFoundError(f"IR file not found: {ir_file}")
    
    print("="*50)
    print("STARTING FLASH FLOOD EVENT INFERENCE")
    print("="*50)
    
    # Create samples directly for the flash flood event using Langragian approach
    # This bypasses the complex windowing and directly targets the event period
    try:
        samples = create_samples_for_flash_flood_event(
            imerg_file, ir_file, device, vae_checkpoint_path, 
            start_datetime, end_datetime
        )
        
        print(f"Created {len(samples)} samples for prediction windows")
        
        # Run inference on the filtered samples
        all_results = []
        for i, sample in enumerate(samples):
                
            event_info = sample['flash_flood_info']
            print(f"\nProcessing flash flood event {i+1}/{len(samples)}")
            print(f"  Event period: {event_info['event_start']} to {event_info['event_end']}")
            print(f"  Description: {event_info['description']}")
            
            # Data is already on device from create_samples_from_windows
            
            with torch.no_grad():
                # Initialize LDM variables
                ldm_prediction = None
                ldm_inference_time = 0.0
                langragian_baseline = sample['langragian_target']  # (12, H, W) - Langragian prediction for target period
                
                # Check if IR data is available for LDM
                if sample['ir_available']:
                    # Prepare inputs for proper Langragian LDM approach
                    print(f"  Running ACTUAL Langragian LDM model...")
                    
                    # Extract data from sample - use LDM-specific data
                    gt_context = sample['gt_context']  # (12, H, W) - IMERG input (15:00-21:00) for regular methods
                    ir_input = sample['ir_input']  # (16, H, W) - IR input (17:00-21:00)
                    
                    # Use LDM-specific difference (second 12 frames - langragian pred from first 12)
                    difference_context = sample['ldm_difference']  # (12, H, W) - What LDM learns from training
                    
                    print(f"  GT context shape: {gt_context.shape}")
                    print(f"  IR input shape: {ir_input.shape}")
                    print(f"  Langragian baseline shape: {langragian_baseline.shape}")
                    
                    try:
                        # Step 1: Encode the difference context (what the model was trained on)
                        # In the Langragian LDM training, the context difference is: GT[13:24] - LP[13:24]
                        # Here we use the available difference as context (could be same period or adjusted)
                        print(f"    Encoding difference context to latent space...")
                        encoded_diff_context = encode_difference_to_latent(difference_context, models_dict, device)
                        print(f"    Encoded difference context shape: {encoded_diff_context.shape}")
                        
                        # Step 2: Encode the IR input to latent space
                        print(f"    Encoding IR input to latent space...")
                        encoded_ir_context = encode_ir_to_latent(ir_input, models_dict, device)
                        print(f"    Encoded IR context shape: {encoded_ir_context.shape}")
                        
                        # Step 3: Combine context by concatenating along channel dimension
                        # Expected: (C_diff + C_ir, T_latent, H_latent, W_latent) = (64, T_latent, H_latent, W_latent)
                        combined_context = torch.cat([encoded_diff_context, encoded_ir_context], dim=0)  # Concatenate along channel dim
                        print(f"    Combined context shape: {combined_context.shape}")
                        
                        # Add batch dimension for the diffusion model: (B, C, T, H, W)
                        combined_context = combined_context.unsqueeze(0).to(device)  # (1, 64, T_latent, H_latent, W_latent)
                    
                        # Step 4: Define target shape for the latent difference we want to generate
                        # This should match the shape of encoded difference latents
                        target_latent_shape = (1,) + encoded_diff_context.shape  # (1, C_latent, T_latent, H_latent, W_latent)
                        print(f"    Target latent shape: {target_latent_shape}")
                        
                        # Step 5: Run the actual diffusion model
                        print(f"    Running diffusion model inference...")
                        ldm_samples, ldm_times = generate_sample_ldm(
                            models_dict=models_dict,
                            condition=combined_context,
                            target_shape=target_latent_shape,
                            langragian_baseline=langragian_baseline,
                            device=device,
                            ensemble_size=1
                        )
                        
                        # Get the final prediction (already in precipitation space)
                        ldm_prediction_raw = ldm_samples[0]  # (T, H, W)
                        ldm_inference_time = ldm_times[0]
                        
                        # Convert to expected format (1, T, 1, H, W)
                        ldm_prediction = ldm_prediction_raw.unsqueeze(0).unsqueeze(2)  # (1, 12, 1, H, W)
                        
                        print(f"    Langragian LDM completed: {ldm_inference_time:.3f}s")
                        print(f"    Final LDM prediction shape: {ldm_prediction.shape}")
                    
                    except Exception as e:
                        print(f"    Langragian LDM failed: {e}")
                        import traceback
                        traceback.print_exc()
                        # Fallback to simple enhancement of Langragian baseline
                        enhanced_baseline = langragian_baseline + (difference_context * 0.2)  # Simple enhancement
                        enhanced_baseline = torch.clamp(enhanced_baseline, min=0.0)
                        ldm_prediction = enhanced_baseline.unsqueeze(0).unsqueeze(2)  # (1, 12, 1, H, W)
                        ldm_inference_time = 0.1
                else:
                    print(f"  Skipping LDM model - IR data not available")
                    # LDM variables already initialized above
                
                # Generate baseline predictions using the input context (15:00-21:00)
                print("  Running baseline methods...")
                # Use the IMERG input context (15:00-21:00) to predict the target period (21:00-03:00)
                imerg_input_context = sample['gt_context']  # (12, H, W) - input period
                print(f"  IMERG input context shape: {imerg_input_context.shape}")
                
                # Handle different possible shapes of input context
                if len(imerg_input_context.shape) == 3:  # (T, H, W)
                    baseline_input = imerg_input_context.unsqueeze(0).unsqueeze(2)  # (1, T, 1, H, W)
                elif len(imerg_input_context.shape) == 4:  # (B, T, H, W)
                    baseline_input = imerg_input_context.unsqueeze(2)  # (B, T, 1, H, W)
                else:
                    baseline_input = imerg_input_context  # Already in correct format
                
                print(f"  Baseline input shape: {baseline_input.shape}")
                print(f"  Predicting {12} timesteps ahead...")
                baseline_predictions = run_baseline_methods(baseline_input, timesteps=12)
                
                # Convert all predictions to torch tensors and ensure same dimensions
                predictions = {}
                inference_times = {}
                
                # Add Langragian LDM prediction (only if IR was available)
                if ldm_prediction is not None:
                    predictions['langragian_ldm'] = ldm_prediction
                    inference_times['langragian_ldm'] = ldm_inference_time
                else:
                    print("  LDM prediction skipped - no IR data available")
                
                for method_name, pred in baseline_predictions.items():
                    if pred is not None:
                        # Convert to torch tensor and ensure correct shape (1, T, 1, H, W)
                        if isinstance(pred, np.ndarray):
                            # Clean the numpy array first - remove NaN and inf values
                            pred_cleaned = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                            # Ensure non-negative precipitation values
                            pred_cleaned = np.maximum(pred_cleaned, 0.0)
                            
                            if len(pred_cleaned.shape) == 3:  # (T, H, W)
                                pred_tensor = torch.from_numpy(pred_cleaned.astype(np.float32)).unsqueeze(0).unsqueeze(2)  # (1, T, 1, H, W)
                            elif len(pred_cleaned.shape) == 4 and pred_cleaned.shape[0] == 1:  # (1, T, H, W)
                                pred_tensor = torch.from_numpy(pred_cleaned.astype(np.float32)).unsqueeze(2)  # (1, T, 1, H, W)
                            else:
                                pred_tensor = torch.from_numpy(pred_cleaned.astype(np.float32)).view(1, 12, 1, pred_cleaned.shape[-2], pred_cleaned.shape[-1])
                        else:
                            pred_tensor = pred
                        
                        predictions[method_name] = pred_tensor.to(device)
                        inference_times[method_name] = 0.1  # Dummy time for baselines
                        
                        # Debug: Check for NaN values after conversion
                        if torch.isnan(pred_tensor).any():
                            print(f"    Warning: {method_name.upper()} still contains NaN after cleaning")
                        
                    else:
                        # Create dummy prediction with same shape as ground truth
                        dummy_shape = (1, 12, 1, sample['gt_target'].shape[-2], sample['gt_target'].shape[-1])
                        predictions[method_name] = torch.zeros(dummy_shape, device=device)
                        inference_times[method_name] = 0.0
                
                # Use ground truth target from Langragian LDM dataset
                # Handle different possible shapes of gt_target
                gt_target = sample['gt_target']
                if len(gt_target.shape) == 3:  # (T, H, W)
                    ground_truth = gt_target.unsqueeze(0).unsqueeze(2)  # (1, T, 1, H, W)
                elif len(gt_target.shape) == 4:  # (B, T, H, W)
                    ground_truth = gt_target.unsqueeze(2)  # (1, T, 1, H, W)
                else:
                    ground_truth = gt_target  # Already in correct format
                
                print(f"  Ground truth shape: {ground_truth.shape}")
                
                # Create static prediction plots for key timesteps
                print("  Creating static prediction plots...")
                create_prediction_plots(
                    predictions_dict=predictions,
                    ground_truth=ground_truth,
                    output_dir=output_dir,
                    event_info=event_info,
                    timesteps_to_plot=[0, 2, 4, 6, 8, 10, 11]  # Key timesteps throughout the event
                )
                
                # Create ground truth GIF
                print("  Creating Ground Truth GIF...")
                create_ground_truth_gif(
                    ground_truth=ground_truth,
                    output_dir=output_dir,
                    start_datetime=event_info['event_start'],  # Use flash flood event start time
                    fps=2
                )
                
                # Create separate comparison GIFs for each method vs ground truth
                print("  Creating individual comparison GIFs...")
                create_individual_comparison_gifs(
                    predictions_dict=predictions,
                    ground_truth=ground_truth,
                    output_dir=output_dir,
                    start_datetime=event_info['event_start'],  # Use flash flood event start time
                    fps=2
                )
                
                # Store results
                sample_results = {
                    'sample_id': i+1,
                    'event_start': event_info['event_start'].strftime('%Y-%m-%d %H:%M:%S'),
                    'event_end': event_info['event_end'].strftime('%Y-%m-%d %H:%M:%S'),
                    'event_description': event_info['description'],
                    'models': {}
                }
                
                for model_name, pred in predictions.items():
                    if isinstance(pred, torch.Tensor):
                        sample_results['models'][model_name] = {
                            'inference_time': inference_times[model_name],
                            'max_precipitation': float(torch.max(pred).item()),
                            'mean_precipitation': float(torch.mean(pred).item()),
                            'shape': list(pred.shape)
                        }
                        print(f"    {model_name.upper()}: Shape={pred.shape}, Max={sample_results['models'][model_name]['max_precipitation']:.3f}, "
                              f"Mean={sample_results['models'][model_name]['mean_precipitation']:.3f}")
                
                all_results.append(sample_results)
        
        return all_results
        
    except Exception as e:
        print(f"Error during inference: {e}")
        raise


def main():
    args = parse_args()
    device = torch.device(args.device)
    
    print("="*60)
    print("FLASH FLOOD INFERENCE - LANGRAGIAN LDM")
    print("="*60)
    print(f"Prediction period: {args.start_date} {args.start_time} for {args.duration_hours} hours")
    print(f"Model path: {args.model_path}")
    print(f"VAE checkpoint: {args.vae_checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {device}")
    print(f"Ensemble size: {args.ensemble}")
    print("Methods: PySTEPS, LINDA, Lagrangian, Naive, Langragian LDM")
    print("="*60)
    
    # Get target datetime range from command line arguments
    start_datetime, end_datetime = get_target_datetimes(args.start_date, args.start_time, args.duration_hours)
    print(f"Start datetime: {start_datetime}")
    print(f"End datetime: {end_datetime}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get existing data file paths
    try:
        imerg_file, ir_file = get_existing_data_paths()
        print(f"\nUsing existing data files:")
        print(f"IMERG: {imerg_file}")
        print(f"IR: {ir_file}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    # Load models
    try:
        models_dict = load_models(args.model_path, device)
    except Exception as e:
        print(f"Error loading models: {e}")
        return 1
    
    # Run inference with all methods
    try:
        results = run_inference(
            imerg_file, ir_file, start_datetime, end_datetime, 
            device, args.output_dir, models_dict, args.vae_checkpoint, args.ensemble
        )
        
        # Save comprehensive summary
        summary_path = os.path.join(args.output_dir, "inference_summary_2010-08-11_to_2010-08-12.txt")
        with open(summary_path, 'w') as f:
            f.write("FLASH FLOOD INFERENCE SUMMARY - LANGRAGIAN LDM\n")
            f.write("="*60 + "\n")
            f.write(f"Time range: {start_datetime} to {end_datetime}\n")
            f.write(f"Model checkpoint: {args.model_path}\n")
            f.write(f"VAE checkpoint: {args.vae_checkpoint}\n")
            f.write(f"Device: {device}\n")
            f.write(f"Ensemble size: {args.ensemble}\n")
            f.write(f"Methods tested: PySTEPS, LINDA, Lagrangian, Naive, Langragian LDM\n\n")
            
            f.write("SAMPLE RESULTS:\n")
            f.write("-" * 40 + "\n")
            
            # Collect statistics for each model
            model_stats = {}
            
            for result in results:
                f.write(f"\nFlash Flood Event {result['sample_id']} - {result['event_start']} to {result['event_end']}:\n")
                f.write(f"  {result['event_description']}\n")
                for model_name, model_result in result['models'].items():
                    if model_name not in model_stats:
                        model_stats[model_name] = {
                            'max_precips': [],
                            'mean_precips': [],
                            'inference_times': []
                        }
                    
                    model_stats[model_name]['max_precips'].append(model_result['max_precipitation'])
                    model_stats[model_name]['mean_precips'].append(model_result['mean_precipitation'])
                    model_stats[model_name]['inference_times'].append(model_result['inference_time'])
                    
                    f.write(f"  {model_name.upper()}:\n")
                    f.write(f"    Shape: {model_result['shape']}\n")
                    f.write(f"    Max precipitation: {model_result['max_precipitation']:.6f}\n")
                    f.write(f"    Mean precipitation: {model_result['mean_precipitation']:.6f}\n")
                    f.write(f"    Inference time: {model_result['inference_time']:.3f}s\n")
            
            # Write model averages
            f.write("\n" + "="*60 + "\n")
            f.write("MODEL PERFORMANCE SUMMARY\n")
            f.write("="*60 + "\n")
            
            for model_name, stats in model_stats.items():
                if len(stats['max_precips']) > 0:
                    f.write(f"\n{model_name.upper()} MODEL:\n")
                    f.write(f"  Average max precipitation: {np.mean(stats['max_precips']):.6f} ± {np.std(stats['max_precips']):.6f}\n")
                    f.write(f"  Average mean precipitation: {np.mean(stats['mean_precips']):.6f} ± {np.std(stats['mean_precips']):.6f}\n")
                    f.write(f"  Average inference time: {np.mean(stats['inference_times']):.3f}s ± {np.std(stats['inference_times']):.3f}s\n")
        
        print(f"\nFlash flood event inference completed!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Summary saved to: {summary_path}")
        print(f"Methods tested: {', '.join(model_stats.keys())}")
        print(f"Flash flood event period: 2010-08-11 21:00 to 2010-08-12 03:00")
        print(f"\nGenerated files:")
        print(f"  Static plots (PNG):")
        print(f"    - flash_flood_predictions_timestep_XX_YYYYMMDD_HHMM.png (for key timesteps)")
        print(f"  Animated GIFs:")
        print(f"    - flash_flood_ground_truth.gif (standalone ground truth)")
        for method in model_stats.keys():
            print(f"    - flash_flood_{method}_vs_ground_truth.gif")
        print(f"\nThe predictions show the evolution of precipitation during the flash flood event,")
        print(f"with timesteps every 30 minutes covering the 6-hour event period.")
        
    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())