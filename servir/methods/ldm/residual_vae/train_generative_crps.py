#!/usr/bin/env python3
"""
Training script for converting deterministic VAE to generative using CRPS fine-tuning.
"""

import os
import sys
import yaml
import logging
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loading.vae_data_loader import VAEDataModule
from models.vae_finetuning import SimpleVAE3D_GenerativeFinetuning
from training.crps_generative_trainer import CRPSGenerativeTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_generative_config() -> dict:
    """Create training configuration for generative conversion."""
    config = {
        'data': {
            'file_path': 'data/imerg_data.h5',
            'batch_size': 4,
            'num_workers': 30,
            'pin_memory': True,
            'shuffle': True,
            'validation_split': 0.1,
            'time_steps': 12,
            'height': 360,
            'width': 516
        },
        'model': {
            'input_channels': 1,
            'latent_dim': 32
        },
        'training': {
            'epochs': 30,
            'learning_rate': 1e-4,           # Original (not used)
            'crps_learning_rate': 1e-5,     # CRPS fine-tuning learning rate
            'grad_clip': 1.0,
            'use_amp': False,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'checkpoint_interval': 1,
            'crps_samples': 9,              # Number of ensemble samples for CRPS
            'debug_crps': False,
            'weight_decay': 0
        },
        'experiment': {
            'name': 'vae_generative_crps',
            'output_dir': 'experiments_generative_crps'
        }
    }
    return config

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Convert deterministic VAE to generative using CRPS')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--pretrained', type=str, required=True, 
                       help='Path to pretrained DETERMINISTIC model checkpoint')
    parser.add_argument('--resume', type=str, help='Path to checkpoint for resuming training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='CRPS learning rate')
    parser.add_argument('--crps_samples', type=int, default=20, help='Number of ensemble samples for CRPS')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = create_generative_config()
    
    # Override config with command line arguments
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['crps_learning_rate'] = args.lr
    if args.crps_samples:
        config['training']['crps_samples'] = args.crps_samples
    
    logger.info("🚀 Starting deterministic-to-generative VAE conversion...")
    logger.info(f"📋 Configuration: {config}")
    
    # Create data loaders
    logger.info("📊 Creating data loaders...")
    data_module = VAEDataModule(
        imerg_filename=config['data']['file_path'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        sequence_length=config['data']['time_steps'],
        image_shape=(config['data']['height'], config['data']['width']),
        normalize_data=True
    )
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    logger.info(f"📈 Training samples: {len(train_loader.dataset)}")
    logger.info(f"📉 Validation samples: {len(val_loader.dataset) if val_loader else 0}")
    
    # Create model
    logger.info("🧠 Creating generative VAE model...")
    model = SimpleVAE3D_GenerativeFinetuning(
        input_channels=config['model']['input_channels'],
        latent_dim=config['model']['latent_dim']
    )
    
    # Create CRPS trainer
    logger.info("🎯 Creating CRPS generative trainer...")
    trainer = CRPSGenerativeTrainer(
        config=config,
        model=model,
        dataloader=train_loader,
        val_loader=val_loader,
        pretrained_checkpoint=args.pretrained,
        resume_checkpoint=args.resume
    )
    
    # Start training
    logger.info("🎯 Starting CRPS generative fine-tuning...")
    logger.info("🔄 Converting deterministic model to generative...")
    try:
        history = trainer.train()
        logger.info("✅ Generative conversion completed successfully!")
        logger.info("🎉 Your deterministic VAE is now GENERATIVE!")
        
        # Save final configuration
        config_path = os.path.join(trainer.exp_dir, "generative_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        logger.info(f"📄 Configuration saved to: {config_path}")
        
    except Exception as e:
        logger.error(f"❌ Generative training failed: {str(e)}")
        raise
    
    logger.info("🎊 Generative conversion completed!")
    logger.info("💡 Your model can now generate multiple diverse outputs!")

if __name__ == "__main__":
    main()
    