# ----------------------- Imports -----------------------
import os
import sys
import torch
import argparse
import datetime
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DataParallel
from torch.amp import autocast
from torch.amp import GradScaler
from tqdm import tqdm
from monai.config import print_config
import matplotlib.pyplot as plt
import csv
from generative.inferers.inferer import LatentDiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

# Add the ldm_data_loader directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ldm_data_loader'))
from langragian_ldm_dataloader import get_langragian_ldm_dataloader


from vae_models.vae_difference_generative import SimpleVAE3D_DifferenceGenerative
from generative.losses.simple_loss import compute_loss

# Create checkpoint directory
checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints_batch_4")
os.makedirs(checkpoint_dir, exist_ok=True)

# -------------------- CSV Loss Logging ------------------
def initialize_loss_csv(csv_path):
    """Initialize CSV file with headers for loss logging"""
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'train_loss', 'val_loss'])

def log_losses_to_csv(csv_path, epoch, train_loss, val_loss=None):
    """Log training and validation losses to CSV file"""
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch, train_loss, val_loss if val_loss is not None else ''])

def load_existing_losses(csv_path):
    """Load existing losses from CSV if resuming training"""
    losses = []
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                losses.append({
                    'epoch': int(row['epoch']),
                    'train_loss': float(row['train_loss']),
                    'val_loss': float(row['val_loss']) if row['val_loss'] else None
                })
    return losses

# -------------------- Argument Parser ------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Latent Diffusion Model Training')
    parser.add_argument('--subset', action='store_true', 
                        help='Use subset of data for testing (default: False)')
    parser.add_argument('--subset-size', type=int, default=100,
                        help='Size of subset when using --subset (default: 50)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate (default: 5e-5)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from (default: None)')
    parser.add_argument('--loss-type', type=str, default='mse',
                        choices=['mse', 'l1', 'l1+l2', 'weighted_mse'],
                        help='Loss type to use (default: mse)')
    parser.add_argument('--precip-weight', type=float, default=4.0,
                        help='Weight factor for precipitation regions (default: 4.0)')
    return parser.parse_args()

# -------------------- Data Loader ----------------------
def get_langragian_data(batch_size=4, device='cuda', use_subset=False, subset_size=300):
    """
    Get data loaders for Langragian LDM training.
    
    Args:
        batch_size: Batch size for dataloaders
        device: Device to use for encoding
        use_subset: Whether to use only a subset of data for testing
        subset_size: Number of samples to use if use_subset=True
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Update these paths to your actual data files
    imerg_file = os.path.join(os.path.dirname(__file__), 'ldm_data_loader', 'imerg_data.h5')
    ir_file = os.path.join(os.path.dirname(__file__), 'ldm_data_loader', 'filled_missing_nan_ir_data.h5')
    
    # VAE checkpoint for difference encoding
    vae_checkpoint = os.path.join(os.path.dirname(__file__), 'vae_models', 'diff_encoder.pth')
    
    # Check if files exist
    if not os.path.exists(imerg_file):
        raise FileNotFoundError(f"IMERG file not found: {imerg_file}")
    if not os.path.exists(ir_file):
        raise FileNotFoundError(f"IR file not found: {ir_file}")
    if not os.path.exists(vae_checkpoint):
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_checkpoint}")
    
    # Create full dataloader
    full_loader = get_langragian_ldm_dataloader(
        imerg_file=imerg_file,
        ir_file=ir_file,
        vae_checkpoint_path=vae_checkpoint,
        batch_size=1,  # Load one at a time for splitting
        shuffle=False,
        num_workers=0,  # Must be 0 for CUDA operations
        device=device,
        ir_start_offset=44,  # 11th hour
        ir_steps=16,
        ir_stride=1
    )
    
    # Get dataset from dataloader
    dataset = full_loader.dataset
    
    # Handle subset selection
    if use_subset:
        total_len = min(subset_size, len(dataset))
        print(f"Using subset of {total_len} samples out of {len(dataset)} total samples")
        
        # Create subset indices
        subset_indices = list(range(total_len))
        dataset = torch.utils.data.Subset(dataset, subset_indices)
    else:
        total_len = len(dataset)
        print(f"Using full dataset with {total_len} samples")
    
    # Split dataset
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len
    
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])
    
    print(f"Dataset split sizes:")
    print(f"  Train set: {len(train_ds)} samples")
    print(f"  Validation set: {len(val_ds)} samples")
    print(f"  Test set: {len(test_ds)} samples")
    
    # No distributed samplers needed for DataParallel
    
    # Create dataloaders with proper collate function
    def collate_fn(batch):
        collated = {}
        keys = batch[0].keys()
        for key in keys:
            collated[key] = torch.stack([item[key] for item in batch])
        return collated
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    
    # Print shapes for verification
    for batch in train_loader:
        print("=== Data shapes ===")
        print("Final Context shape:", batch['context'].shape)
        print("Target shape:", batch['target'].shape)
        print("\n=== Original data shapes for VAE ===")
        print("GT Input shape:", batch['gt_context'].shape)
        print("GT Target shape:", batch['gt_target'].shape)
        print("Langragian Input shape:", batch['langragian_context'].shape)
        print("Langragian Target shape:", batch['langragian_target'].shape)
        print("IR Raw shape:", batch['ir_raw'].shape)
       
        break
    
    return train_loader, val_loader, test_loader


# ------------------- GPU Setup -------------------------
def setup_gpu():
    """
    Setup for multi-GPU training using DataParallel.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    
    print(f"Using device: {device}")
    print(f"Available GPUs: {num_gpus}")
    
    return device, num_gpus


# ------------------ Main Training Loop ----------------------
def main(args=None):
    # Parse arguments if not provided
    if args is None:
        args = parse_args()
    
    # Setup GPU
    device, num_gpus = setup_gpu()
    
    # Clear GPU memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory cleared before training")
    
    # Print training configuration
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION")
    print("="*50)
    print(f"Mode: {'Subset testing' if args.subset else 'Full training'}")
    if args.subset:
        print(f"Subset size: {args.subset_size} samples")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Using {num_gpus} GPU(s)")
    print("="*50 + "\n")
    
    # Load generative difference VAE model for decoding
    vae_diff_model = SimpleVAE3D_DifferenceGenerative(input_channels=1, latent_dim=32).to(device)
    
    # Load VAE checkpoint if available
    vae_diff_checkpoint = os.path.join(os.path.dirname(__file__), 'vae_models', 'diff_encoder.pth')
    if os.path.exists(vae_diff_checkpoint):
        checkpoint = torch.load(vae_diff_checkpoint, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            vae_diff_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded difference VAE from {vae_diff_checkpoint}")
            print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        else:
            # Try loading as direct state dict
            vae_diff_model.load_state_dict(checkpoint)
            print(f"Loaded difference VAE state dict from {vae_diff_checkpoint}")
    else:
        print("Warning: Difference VAE checkpoint not found, using random initialization")
    
    # Set VAE to eval mode
    vae_diff_model.eval()
    
    # Get data loaders for Langragian LDM training
    train_loader, val_loader, test_loader = get_langragian_data(
        batch_size=args.batch_size,
        device=device,
        use_subset=args.subset,
        subset_size=args.subset_size
    )
    
    # Initialize U-Net for latent diffusion with difference encoding
    unet = DiffusionModelUNet(
        with_conditioning=True,
        cross_attention_dim=64,  # Context has both difference + IR encoded features
        spatial_dims=3,
        in_channels=32,  # Target latent channels from difference VAE
        out_channels=32,  # Target latent channels from difference VAE
        num_res_blocks=6,
        num_channels=(128, 256, 512, 512),        # 4 levels
        attention_levels=(False, True, True, True), # 4 levels
        num_head_channels=(0, 256, 512, 512),     # 4 levels
    ).to(device)
    
    # Use DataParallel for multi-GPU training
    if num_gpus > 1:
        model = DataParallel(unet)
        print(f"Using DataParallel with {num_gpus} GPUs")
    else:
        model = unet
        print("Using single GPU")
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
        beta_start=0.0001,
        beta_end=0.02
    )
    
    inferer = LatentDiffusionInferer(scheduler)
    scaler = GradScaler()
    
    # Simple loss function selection
    print(f"\nUsing loss type: {args.loss_type}")
    if args.loss_type == 'weighted_mse':
        print(f"Precipitation weight factor: {args.precip_weight}")
    print("")
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming training from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            
            # Load model state (handle DataParallel wrapper)
            if num_gpus > 1:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Get starting epoch
            start_epoch = checkpoint['epoch'] + 1
            
            # Set random seed based on epoch
            torch.manual_seed(42 + start_epoch * 1000)
            np.random.seed(42 + start_epoch * 1000)
            torch.cuda.manual_seed_all(42 + start_epoch * 1000)
            
            # Load and override loss configuration from checkpoint if available
            if 'loss_type' in checkpoint:
                args.loss_type = checkpoint['loss_type']
                print(f"Loaded loss type from checkpoint: {args.loss_type}")
            
            if 'precip_weight' in checkpoint:
                args.precip_weight = checkpoint['precip_weight']
                print(f"Loaded precipitation weight from checkpoint: {args.precip_weight}")
            
            # Load previous loss if available
            if 'loss' in checkpoint:
                print(f"Previous training loss: {checkpoint['loss']:.6f}")
            
            print(f"Resuming from epoch {start_epoch}")
            print(f"Using loss type: {args.loss_type}")
            if args.loss_type == 'weighted_mse':
                print(f"Precipitation weight factor: {args.precip_weight}")
        else:
            print(f"Warning: Resume checkpoint not found at {args.resume}")
            print("Starting training from scratch")
    
    # Initialize CSV loss logging
    csv_path = os.path.join(checkpoint_dir, "training_losses.csv")
    if not args.resume or not os.path.exists(csv_path):
        initialize_loss_csv(csv_path)
        print(f"Initialized loss CSV at: {csv_path}")
    else:
        print(f"Resuming with existing loss CSV: {csv_path}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=80)
        
        for step, batch in enumerate(progress_bar):
            
            # Extract context and target from Langragian LDM dataset
            context = batch['context'].to(device)  # Combined encoded differences + IR
            target = batch['target'].to(device)    # Encoded difference target
            
            # Data is already in (B, T, C, H, W) format, permute to (B, C, T, H, W)
            context = context.permute(0, 2, 1, 3, 4)
            target = target.permute(0, 2, 1, 3, 4)
            
            # No original data available for loss weighting in difference encoding
            imerg_output_original = None
            
            # Generate noise and timesteps
            noise = torch.randn_like(target).to(device)
            timesteps = torch.randint(
                0, scheduler.num_train_timesteps, 
                (target.shape[0],), device=device
            ).long()
            
            # Forward pass
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", dtype=torch.float16):
                # Simple forward diffusion process
                noisy_target = scheduler.add_noise(target, noise, timesteps)
                
                # Predict noise
                noise_pred = model(
                    x=noisy_target,
                    timesteps=timesteps,
                    context=context
                )
                
                # Calculate loss using simple loss function from Stage1
                # For difference encoding, we don't have original precipitation data for weighting
                loss = compute_loss(
                    noise_pred.float(), 
                    noise.float(),
                    loss_type=args.loss_type,
                    original_data=None,  # No original data available for difference encoding
                    precip_weight=args.precip_weight,
                    precip_threshold=0.1
                )
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Track loss
            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss / (step + 1))
            
            # Clear GPU cache between batches to prevent memory fragmentation
            if step % 10 == 0:  # Clear every 10 steps to balance performance vs memory
                torch.cuda.empty_cache()
        
        # Calculate average training loss
        avg_train_loss = total_loss / len(train_loader)
        
        # Clear GPU cache after each epoch
        torch.cuda.empty_cache()
        
        # Save checkpoint
        save_path = os.path.join(checkpoint_dir, f"ldm_epoch_{epoch}.pt")
        if num_gpus > 1:
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
            
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
            'loss_type': args.loss_type,
            'precip_weight': args.precip_weight
        }, save_path)
        print(f"Epoch {epoch} completed. Model saved to {save_path}")
        
        # Validation (every epoch)
        if True:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for val_step, batch in enumerate(val_loader):
                    # Extract context and target from Langragian LDM dataset
                    context = batch['context'].to(device)  # Combined encoded differences + IR
                    target = batch['target'].to(device)    # Encoded difference target
                    
                    # Data is already in (B, T, C, H, W) format, permute to (B, C, T, H, W)
                    context = context.permute(0, 2, 1, 3, 4)
                    target = target.permute(0, 2, 1, 3, 4)
                    
                    # No original data available for loss weighting in difference encoding
                    imerg_output_original = None
                    
                    # Generate noise and timesteps
                    noise = torch.randn_like(target).to(device)
                    timesteps = torch.randint(
                        0, scheduler.num_train_timesteps, 
                        (target.shape[0],), device=device
                    ).long()
                    
                    with autocast(device_type="cuda", dtype=torch.float16):
                        # Simple forward diffusion process
                        noisy_target = scheduler.add_noise(target, noise, timesteps)
                        
                        # Predict noise
                        noise_pred = model(
                            x=noisy_target,
                            timesteps=timesteps,
                            context=context
                        )
                        
                        # Calculate loss using simple loss function from Stage1
                        # For difference encoding, we don't have original precipitation data for weighting
                        loss_val = compute_loss(
                            noise_pred.float(), 
                            noise.float(),
                            loss_type=args.loss_type,
                            original_data=None,  # No original data available for difference encoding
                            precip_weight=args.precip_weight,
                            precip_threshold=0.1
                        )
                    
                    # Track validation loss
                    val_loss += loss_val.item()
                    
                    # Clear GPU cache periodically during validation
                    if val_step % 5 == 0:  # More frequent clearing
                        torch.cuda.empty_cache()
        
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch} Validation Loss: {avg_val_loss:.6f}")
        
        # Log losses to CSV (every epoch)
        log_losses_to_csv(csv_path, epoch, avg_train_loss, avg_val_loss)
        
        # Generate visualizations every 2 epochs for GIF creation
        if epoch % 2 == 0:  # Generate every 2 epochs
            with torch.no_grad():
                # Get a random validation sample to reconstruct the full data flow
                random_idx = torch.randint(0, len(val_loader.dataset), (1,)).item()
                sample = val_loader.dataset[random_idx]
                
                # Move sample to device and add batch dimension
                sample_context = sample['context'].unsqueeze(0).to(device)
                sample_target = sample['target'].unsqueeze(0).to(device)
                
                # Predict the difference using the trained model
                # Use deterministic sampling (no noise) for consistent visualization
                sample_context_formatted = sample_context.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
                sample_target_formatted = sample_target.permute(0, 2, 1, 3, 4)    # (B, C, T, H, W)
                
                # Generate prediction using DDIM sampling (faster and deterministic)
                latent_sample = torch.randn_like(sample_target_formatted).to(device)
                
                # Faster sampling with fewer steps
                num_inference_steps = 50
                step_size = scheduler.num_train_timesteps // num_inference_steps
                
                for i, t in enumerate(range(scheduler.num_train_timesteps - 1, -1, -step_size)):
                    t_tensor = torch.tensor([t], device=device).long()
                    
                    # Predict noise
                    noise_pred = model(
                        x=latent_sample,
                        timesteps=t_tensor,
                        context=sample_context_formatted
                    )
                    
                    # Remove noise (simple DDIM step)
                    alpha_t = scheduler.alphas_cumprod[t]
                    alpha_prev = scheduler.alphas_cumprod[max(0, t - step_size)]
                    
                    pred_x0 = (latent_sample - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
                    latent_sample = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * noise_pred
                
                # Decode both predicted and ground truth differences using the VAE
                # latent_sample is in (B, C, T, H, W) format - VAE expects this format
                pred_diff_latent = latent_sample  # Already in correct format (B, C, T, H, W)
                
                # sample_target is in (B, T, C, H, W), convert to (B, C, T, H, W) for VAE
                gt_diff_latent = sample_target.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W) -> (B, C, T, H, W)
                
                try:
                    # Decode only the predicted differences using the VAE
                    pred_diff_decoded = vae_diff_model.decode(pred_diff_latent)
                    
                    # Now we need to reconstruct the Langragian predictions and GT to show final precipitation
                    # We'll need to get the original data and recreate the Langragian predictions
                    
                    # Get original data directly from the validation sample
                    val_sample = val_loader.dataset[random_idx]
                    
                    # Extract original data that's now available in the dataloader
                    gt_precipitation = val_sample['gt_target'].numpy()  # GT[25:36]
                    langragian_prediction = val_sample['langragian_target'].numpy()  # LP[25:36]
                    predicted_differences = pred_diff_decoded[0].cpu().numpy()  # Our model's predicted corrections
                    
                    # Final corrected prediction = Langragian + Predicted_differences
                    corrected_prediction = langragian_prediction + predicted_differences[:, 0, :, :]  # Remove channel dim
                    
                    # Create GIF frames for all 12 timesteps
                    n_timesteps = min(12, gt_precipitation.shape[0])
                    
                    # Create individual frames for GIF
                    for t in range(n_timesteps):
                        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                        
                        # Ground Truth Precipitation
                        im1 = axes[0].imshow(gt_precipitation[t], 
                                           cmap='YlGnBu', vmin=0, vmax=20)
                        axes[0].set_title(f'Ground Truth (t={t})', fontsize=14)
                        axes[0].axis('off')
                        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='mm/hr')
                        
                        # Corrected Prediction (Langragian + Predicted Differences)
                        im2 = axes[1].imshow(corrected_prediction[t], 
                                           cmap='YlGnBu', vmin=0, vmax=20)
                        axes[1].set_title(f'Corrected Prediction (t={t})', fontsize=14)
                        axes[1].axis('off')
                        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='mm/hr')
                        
                        # Raw Langragian Prediction (for comparison)
                        im3 = axes[2].imshow(langragian_prediction[t], 
                                           cmap='YlGnBu', vmin=0, vmax=20)
                        axes[2].set_title(f'Raw Langragian (t={t})', fontsize=14)
                        axes[2].axis('off')
                        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04, label='mm/hr')
                        
                        # Add epoch and loss info
                        fig.suptitle(f'Epoch {epoch:02d} - Timestep {t:02d}/12 - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}', 
                                    fontsize=16, y=0.95)
                        
                        plt.tight_layout()
                        
                        # Save frame for GIF
                        frame_path = os.path.join(checkpoint_dir, f"gif_frames", f"epoch_{epoch:03d}_timestep_{t:02d}.png")
                        os.makedirs(os.path.dirname(frame_path), exist_ok=True)
                        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
                        plt.close()
                        
                    # Also create a summary comparison showing improvement
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    
                    # Show first and last timesteps
                    for row, t in enumerate([0, 11]):
                        # GT
                        im1 = axes[row, 0].imshow(gt_precipitation[t], cmap='YlGnBu', vmin=0, vmax=20)
                        axes[row, 0].set_title(f'GT (t={t})')
                        axes[row, 0].axis('off')
                        
                        # Corrected
                        im2 = axes[row, 1].imshow(corrected_prediction[t], cmap='YlGnBu', vmin=0, vmax=20)
                        axes[row, 1].set_title(f'Corrected (t={t})')
                        axes[row, 1].axis('off')
                        
                        # Langragian only
                        im3 = axes[row, 2].imshow(langragian_prediction[t], cmap='YlGnBu', vmin=0, vmax=20)
                        axes[row, 2].set_title(f'Langragian (t={t})')
                        axes[row, 2].axis('off')
                    
                    fig.suptitle(f'Epoch {epoch:02d} Summary - GT vs Corrected vs Langragian', fontsize=16)
                    plt.tight_layout()
                    summary_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}_summary.png")
                    plt.savefig(summary_path, dpi=100, bbox_inches='tight')
                    plt.close()
                    
                    # Create GIF from this epoch's frames
                    try:
                        from PIL import Image
                        import glob
                        
                        # Collect all frames for this epoch
                        frame_pattern = os.path.join(checkpoint_dir, f"gif_frames", f"epoch_{epoch:03d}_timestep_*.png")
                        frame_files = sorted(glob.glob(frame_pattern))
                        
                        if frame_files:
                            images = []
                            for frame_file in frame_files:
                                img = Image.open(frame_file)
                                images.append(img)
                            
                            # Save GIF for this epoch
                            gif_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}_animation.gif")
                            images[0].save(
                                gif_path,
                                save_all=True,
                                append_images=images[1:],
                                duration=500,  # 500ms per frame
                                loop=0
                            )
                            print(f"  -> Created GIF: epoch_{epoch:03d}_animation.gif")
                            
                    except ImportError:
                        print("  -> PIL not available, install with: pip install Pillow")
                    
                except Exception as e:
                    print(f"  -> VAE decoding failed: {e}")
                    print(f"  -> Dataset type: {type(val_loader.dataset)}")
                    if hasattr(val_loader.dataset, 'dataset'):
                        print(f"  -> Actual dataset type: {type(val_loader.dataset.dataset)}")
                    print("  -> Using latent space visualization instead")
                    
                    # Fallback to latent space visualization
                    n_timesteps = min(12, pred_diff_latent.shape[1])
                    for t in range(n_timesteps):
                        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                        
                        # GT latent
                        im1 = axes[0].imshow(gt_diff_latent[0, t, 0].cpu().numpy(), 
                                           cmap='RdBu_r', vmin=-2, vmax=2)
                        axes[0].set_title(f'GT Latent (t={t})')
                        axes[0].axis('off')
                        
                        # Pred latent
                        im2 = axes[1].imshow(pred_diff_latent[0, t, 0].cpu().numpy(), 
                                           cmap='RdBu_r', vmin=-2, vmax=2)
                        axes[1].set_title(f'Pred Latent (t={t})')
                        axes[1].axis('off')
                        
                        fig.suptitle(f'Epoch {epoch:02d} - Timestep {t:02d} (Latent Space)', fontsize=14)
                        plt.tight_layout()
                        
                        # Save frame
                        frame_path = os.path.join(checkpoint_dir, f"latent_frames", f"epoch_{epoch:03d}_timestep_{t:02d}.png")
                        os.makedirs(os.path.dirname(frame_path), exist_ok=True)
                        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
                        plt.close()
                
                print(f"Epoch {epoch}: Generated 12-timestep visualization frames")
    
    print(f"\nTraining completed! Check {checkpoint_dir} for:")
    print("  - Individual epoch GIFs: epoch_XXX_animation.gif")


# ------------------ Entry Point --------------------
if __name__ == "__main__":
    # Parse arguments first
    args = parse_args()
    
    print("Starting training...")
    main(args=args)
    print("Training completed.")