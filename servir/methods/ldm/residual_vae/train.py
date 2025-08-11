from src.models.vae import SimpleVAE3D
from src.training.trainer import VAETrainer
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import logging
from src.data_loading.vae_data_loader import VAEDataModule
from torch.utils.data import Subset
from src.extrapolation_methods.langragian_persistance import langragian_persistance
import numpy as np
import glob

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    h5_dataset_location='data/imerg_data.h5'

    # Training configuration (matching encoder_decoder_imerg_trial_3 style)
    config = {
        'data': {
            'batch_size': 8,
            'input_shape': [1, 12, 360, 516],  # [channels, depth, height, width]
            'num_workers': 40,
            'pin_memory': True,
            'persistent_workers': True
        },
        'model': {
            'latent_dim': 32
        },
        'training': {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'learning_rate': 1e-5,
            'epochs': 100,
            'beta': 1.5,
            'use_amp': True,
            'grad_clip': 1.0,
            'checkpoint_interval': 1
        },
        'experiment': {
            'name': 'vae_diff_training_lr_1e-5',
            'output_dir': 'experiments_lr_1e-5'
        }
    }

    # Ensure output directory exists
    os.makedirs(config['experiment']['output_dir'], exist_ok=True)

    # VAE data loader setup
    data_provider = VAEDataModule(
        sequence_length=12,
        imerg_filename=h5_dataset_location,
        batch_size=config['data']['batch_size'],
        image_shape=(360,516),
        normalize_data=False,
        train_split=0.8,
        val_split=0.1
    )

    test_data_loader = data_provider.test_dataloader()
    train_data_loader = data_provider.train_dataloader()
    val_data_loader = data_provider.val_dataloader()

    train_dataset_size = len(data_provider.train_sequences)
    val_dataset_size = len(data_provider.val_sequences)
    test_dataset_size = len(data_provider.test_sequences)

    logger.info(f"Training dataset size: {train_dataset_size}")
    logger.info(f"Validation dataset size: {val_dataset_size}")
    logger.info(f"Test dataset size: {test_dataset_size}")

    # Initialize VAE model (same as encoder_decoder_imerg_trial_3)
    model = SimpleVAE3D(
        input_channels=config['data']['input_shape'][0],
        latent_dim=config['model']['latent_dim']
    )

    # Custom dataset for differences (truly on-the-fly computation)
    class DifferenceDataset(torch.utils.data.Dataset):
        def __init__(self, original_dataset, window_size=12):
            self.original_dataset = original_dataset
            self.window_size = window_size
            # We need pairs, so length is original length - 1
            self.length = len(original_dataset) - 1
            
            logger.info(f"Initialized on-the-fly difference dataset with {self.length} sample pairs")
        
        def __len__(self):
            return self.length
        
        def __getitem__(self, idx):
            try:
                # Get consecutive samples from original dataset
                current_data = self.original_dataset[idx]
                next_data = self.original_dataset[idx + 1]
                
                # Convert to numpy if needed
                if hasattr(current_data, 'numpy'):
                    current_data = current_data.numpy()
                    next_data = next_data.numpy()
                
                # Handle the tensor format - check dimensions
                if len(current_data.shape) == 4:  # [time, channels, height, width]
                    current_sample = current_data[:, 0, :, :]  # [12, 360, 516]
                    ground_truth_next = next_data[:, 0, :, :]  # [12, 360, 516]
                else:  # [batch, time, channels, height, width]
                    current_sample = current_data[0, :, 0, :, :]  # [12, 360, 516]
                    ground_truth_next = next_data[0, :, 0, :, :]  # [12, 360, 516]
                
                # Compute Lagrangian persistence on-the-fly
                lagrangian_pred = langragian_persistance(current_sample, self.window_size)
                
                # Compute difference (residual)
                difference = ground_truth_next - lagrangian_pred
                
                # Return with correct dimensions: [channels, time, height, width]
                return torch.tensor(difference, dtype=torch.float32).unsqueeze(0)  # Shape: [1, 12, 360, 516]
                
            except Exception as e:
                logger.warning(f"Error computing difference for sample {idx}: {e}")
                # Return zero tensor as fallback with correct dimensions
                return torch.zeros((1, self.window_size, 360, 516), dtype=torch.float32)

    # Create simple datasets from sequences
    from src.data_loading.vae_data_loader import SimpleSequenceDataset
    train_dataset = SimpleSequenceDataset(data_provider.train_sequences)
    val_dataset = SimpleSequenceDataset(data_provider.val_sequences)
    
    # Create difference datasets
    logger.info("Creating difference datasets...")
    train_diff_dataset = DifferenceDataset(train_dataset)
    val_diff_dataset = DifferenceDataset(val_dataset)

    # Optional: Create subset for testing on smaller dataset
    USE_SUBSET = False # Set to False to use full dataset
    SUBSET_SIZE = 100  # Number of samples for testing

    if USE_SUBSET and len(train_diff_dataset) > SUBSET_SIZE:
        logger.info(f"Using subset of {SUBSET_SIZE} samples for training (out of {len(train_diff_dataset)})")
        train_subset = Subset(train_diff_dataset, range(min(SUBSET_SIZE, len(train_diff_dataset))))
        train_diff_loader = DataLoader(train_subset, batch_size=config['data']['batch_size'], shuffle=True)
        
        val_subset_size = min(20, len(val_diff_dataset))  # Use 20 samples for validation
        val_subset = Subset(val_diff_dataset, range(val_subset_size))
        val_diff_loader = DataLoader(val_subset, batch_size=config['data']['batch_size'], shuffle=False)
        
        logger.info(f"Training subset size: {len(train_subset)}")
        logger.info(f"Validation subset size: {len(val_subset)}")
    else:
        logger.info(f"Using full dataset - Training: {len(train_diff_dataset)}, Validation: {len(val_diff_dataset)}")
        train_diff_loader = DataLoader(train_diff_dataset, batch_size=config['data']['batch_size'], shuffle=True)
        val_diff_loader = DataLoader(val_diff_dataset, batch_size=config['data']['batch_size'], shuffle=False)

    # Resume training configuration
    RESUME_TRAINING = False
    RESUME_CHECKPOINT_PATH = None
    # Example checkpoint path (update with your actual path):
    # RESUME_CHECKPOINT_PATH = "experiments/vae_diff_training_20250708_123456/checkpoints/checkpoint_epoch_10.pth"

    # Auto-find latest checkpoint (optional)
    AUTO_RESUME_LATEST = False  # Set to True to automatically resume from latest checkpoint
    if AUTO_RESUME_LATEST:
        checkpoint_pattern = "experiments/vae_diff_training_*/checkpoints/checkpoint_epoch_*.pth"
        checkpoints = glob.glob(checkpoint_pattern)
        if checkpoints:
            # Sort by modification time and get the latest
            latest_checkpoint = max(checkpoints, key=os.path.getmtime)
            RESUME_TRAINING = True
            RESUME_CHECKPOINT_PATH = latest_checkpoint
            logger.info(f"Auto-resuming from latest checkpoint: {latest_checkpoint}")

    # Initialize trainer
    trainer = VAETrainer(
        config=config,
        model=model,
        dataloader=train_diff_loader,
        val_loader=val_diff_loader,
        resume_checkpoint=RESUME_CHECKPOINT_PATH if RESUME_TRAINING else None
    )

    # Start training
    logger.info("Starting VAE training on differences...")
    history = trainer.train()

    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()