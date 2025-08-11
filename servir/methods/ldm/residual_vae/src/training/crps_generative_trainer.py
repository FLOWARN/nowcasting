# src/training/crps_generative_trainer.py
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from datetime import datetime
import numpy as np
import csv
from typing import Dict, Any, Optional, Tuple

from training.trainer import VAETrainer
from utils.crps_loss import CRPSLoss
from models.vae_finetuning import SimpleVAE3D_GenerativeFinetuning

logger = logging.getLogger(__name__)

class CRPSGenerativeTrainer(VAETrainer):
    """CRPS-based trainer for converting deterministic VAE to generative using decoder fine-tuning."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        model: nn.Module,
        dataloader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        pretrained_checkpoint: Optional[str] = None,
        resume_checkpoint: Optional[str] = None
    ):
        """
        Initialize CRPS generative trainer.
        
        Args:
            config: Training configuration dictionary
            model: VAE model to train
            dataloader: DataLoader for training data
            val_loader: DataLoader for validation data
            pretrained_checkpoint: Path to pretrained DETERMINISTIC model
            resume_checkpoint: Path to checkpoint for resuming training
        """
        # Initialize parent class
        super().__init__(config, model, dataloader, val_loader, resume_checkpoint)
        
        # Initialize CRPS loss for generative training
        self.crps_loss = CRPSLoss(
            num_samples=config['training'].get('crps_samples', 20),
            reduction='mean'
        )
        
        # Load pretrained DETERMINISTIC model and freeze encoder
        if pretrained_checkpoint and isinstance(model, SimpleVAE3D_GenerativeFinetuning):
            logger.info(f"Loading pretrained DETERMINISTIC model from: {pretrained_checkpoint}")
            model.load_pretrained_encoder(pretrained_checkpoint)
            logger.info("✓ Loaded deterministic model and froze encoder - ready for generative fine-tuning!")
        elif pretrained_checkpoint:
            logger.warning("Pretrained checkpoint specified but model is not SimpleVAE3D_GenerativeFinetuning")
            
        # Print parameter status
        if hasattr(model, 'print_parameter_status'):
            model.print_parameter_status()

    def _setup_training(self) -> None:
        """Initialize optimizer for decoder-only CRPS training."""
        # Get trainable parameters (decoder only - encoder is frozen)
        if hasattr(self.model, 'get_trainable_parameters'):
            trainable_params = list(self.model.get_trainable_parameters())
        else:
            trainable_params = list(self.model.parameters())
            
        # Use much lower learning rate for CRPS fine-tuning
        lr = self.config['training'].get('crps_learning_rate', 1e-5)
        
        self.optimizer = torch.optim.Adam(
            trainable_params,
            lr=lr,
            betas=(0.9, 0.999),  # Different betas for CRPS
            weight_decay=self.config['training'].get('weight_decay', 0)
        )
        
        logger.info(f"✓ CRPS Optimizer initialized with {len(trainable_params)} trainable parameters, lr={lr}")
        
        # Enable mixed precision training if specified
        self.scaler = torch.amp.GradScaler('cuda',
            enabled=self.config['training'].get('use_amp', False)
        )
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
            
        self.model.to(self.device)

    def crps_generative_loss(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        ensemble: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute PURE CRPS loss for generative training (no KL loss).
        
        Args:
            x: Original input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            ensemble: Precomputed ensemble predictions (optional)
            
        Returns:
            Tuple of (crps_loss, crps_loss) - same value twice for compatibility
        """
        # Compute CRPS loss using ensemble or generate on-the-fly
        if ensemble is not None:
            # Use precomputed ensemble
            crps_loss = self.crps_loss.empirical_crps_vectorized(ensemble, x).mean()
        else:
            # Generate ensemble on the fly
            decoder_fn = self.model.module.decode if isinstance(self.model, nn.DataParallel) else self.model.decode
            crps_loss = self.crps_loss(mu, logvar, decoder_fn, x)
        
        # Return CRPS loss (no KL loss - we want to maximize diversity)
        return crps_loss, crps_loss

    def train(self) -> Dict[str, list]:
        """Main CRPS generative training loop."""
        history = {
            'crps_loss': [],
            'val_crps_loss': []
        }
       
        # Create history files
        history_file_path = os.path.join(self.exp_dir, "crps_generative_history.csv")
        iteration_history_file_path = os.path.join(self.exp_dir, "crps_generative_iteration_history.csv")
    
        with open(history_file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['epoch', 'crps_loss', 'val_crps_loss'])
        
        with open(iteration_history_file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['epoch', 'iteration', 'crps_loss'])
        
        logger.info("🚀 Starting CRPS generative fine-tuning...")
        logger.info("📋 Converting deterministic VAE to generative model...")
        
        try:
            for epoch in range(self.start_epoch, self.config['training']['epochs']):
                self.current_epoch = epoch
                self.model.train()
                epoch_crps_loss = 0
                
                # Training phase
                with tqdm(self.dataloader, desc=f"🎯 CRPS Gen Epoch {epoch+1}/{self.config['training']['epochs']}") as pbar:
                    for iteration, batch in enumerate(pbar):
                        batch = batch.to(self.device)
                        
                        # Clear cache before forward pass
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Forward pass with mixed precision
                        with torch.amp.autocast('cuda',
                            enabled=self.config['training'].get('use_amp', False)
                        ):
                            # Use ensemble forward for CRPS training
                            if hasattr(self.model, 'forward_with_ensemble') or hasattr(self.model.module, 'forward_with_ensemble'):
                                if isinstance(self.model, nn.DataParallel):
                                    recon_batch, mu, logvar, ensemble = self.model.module.forward_with_ensemble(
                                        batch, num_samples=self.config['training'].get('crps_samples', 20)
                                    )
                                else:
                                    recon_batch, mu, logvar, ensemble = self.model.forward_with_ensemble(
                                        batch, num_samples=self.config['training'].get('crps_samples', 20)
                                    )
                                
                                # Compute PURE CRPS loss with precomputed ensemble
                                crps_loss, _ = self.crps_generative_loss(batch, mu, logvar, ensemble)
                            else:
                                # Fallback to regular forward pass
                                recon_batch, mu, logvar = self.model(batch)
                                crps_loss, _ = self.crps_generative_loss(batch, mu, logvar, None)
                        
                        # Backward pass
                        self.optimizer.zero_grad()
                        self.scaler.scale(crps_loss).backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['training'].get('grad_clip', 1.0)
                        )
                        
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        
                        # Update metrics
                        epoch_crps_loss += crps_loss.item()
                        
                        # Log iteration data
                        with open(iteration_history_file_path, mode='a', newline='') as csv_file:
                            writer = csv.writer(csv_file)
                            writer.writerow([epoch + 1, iteration + 1, crps_loss.item()])
                        
                        pbar.set_postfix({
                            'crps_loss': crps_loss.item(),
                            'mode': 'generative'
                        })
                
                # Calculate epoch metrics
                dataset_size = len(self.dataloader.dataset)
                avg_crps_loss = epoch_crps_loss / dataset_size
                
                history['crps_loss'].append(avg_crps_loss)
                
                # Validation
                val_crps_loss = self._validate_crps_generative() if self.val_loader else None
                if val_crps_loss is not None:
                    history['val_crps_loss'].append(val_crps_loss)
                    logger.info(f"🎯 CRPS Gen Epoch {epoch+1}: CRPS={avg_crps_loss:.6f}, Val_CRPS={val_crps_loss:.6f}")
                else:
                    logger.info(f"🎯 CRPS Gen Epoch {epoch+1}: CRPS={avg_crps_loss:.6f}")
                
                # Save epoch data
                with open(history_file_path, mode='a', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([epoch + 1, avg_crps_loss, val_crps_loss if val_crps_loss is not None else 'N/A'])
                
                # Checkpointing
                current_loss = val_crps_loss if val_crps_loss is not None else avg_crps_loss
                is_best = current_loss < self.best_loss
                if is_best:
                    self.best_loss = current_loss
                
                # Save checkpoint
                if (epoch + 1) % self.config['training'].get('checkpoint_interval', 5) == 0:
                    self._save_checkpoint(epoch, current_loss, is_best=is_best)
                
        except Exception as e:
            logger.error(f"CRPS generative training failed at epoch {epoch}: {str(e)}")
            self._save_checkpoint(epoch, self.best_loss, emergency=True)
            raise
        
        # Save final model
        self._save_checkpoint(epoch, self.best_loss)
        logger.info("✅ CRPS generative training completed successfully!")
        logger.info("🎉 Your deterministic VAE is now GENERATIVE!")
        return history

    def _validate_crps_generative(self) -> float:
        """Run CRPS validation for generative model."""
        if self.val_loader is None:
            return 0.0
            
        self.model.eval()
        total_crps_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                
                # Get latent parameters
                if isinstance(self.model, nn.DataParallel):
                    mu, logvar = self.model.module.encode(batch)
                    decoder_fn = self.model.module.decode
                else:
                    mu, logvar = self.model.encode(batch)
                    decoder_fn = self.model.decode
                
                # Compute CRPS loss
                crps_loss = self.crps_loss.evaluate(mu, logvar, decoder_fn, batch)
                total_crps_loss += crps_loss.item()
        
        avg_crps_loss = total_crps_loss / len(self.val_loader.dataset)
        logger.info(f"🎯 CRPS Generative Validation: {avg_crps_loss:.6f}")
        return avg_crps_loss