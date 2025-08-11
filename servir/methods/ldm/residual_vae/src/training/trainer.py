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
from typing import Dict, Any, Optional, Tuple  # Add Tuple to the imports
logger = logging.getLogger(__name__)

class VAETrainer:
    """Handles training of 3D VAE model with checkpointing and visualization."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        model: nn.Module,
        dataloader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        resume_checkpoint: Optional[str] = None
    ):
        """
        Initialize trainer with configuration, model, and data.
        
        Args:
            config: Training configuration dictionary
            model: VAE model to train
            dataloader: DataLoader for training data
            visualizer: Visualization handler
            resume_checkpoint: Path to checkpoint for resuming training
        """
        self.config = config
        self.model = model
        self.dataloader = dataloader
        self.val_loader = val_loader
        
        # Training state
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.current_epoch = 0
        
        # Setup device and training tools
        self.device = torch.device(config['training']['device'])
        self._setup_training()
        self._setup_output_dirs()
        
        # Handle resume if requested
        if resume_checkpoint:
            self._load_checkpoint(resume_checkpoint)

    def _setup_training(self) -> None:
        """Initialize optimizer and multi-GPU training."""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            betas=(0.8,0.999)
        )
        
        # Enable mixed precision training if specified
        self.scaler = torch.amp.GradScaler('cuda',
            enabled=self.config['training'].get('use_amp', False)
        )
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
            # in our gpu cluster we cant use more than 2 gpus for dataparallel idk why !!
            
        self.model.to(self.device)

    def _setup_output_dirs(self) -> None:
        """Create experiment directory structure."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(
            self.config['experiment']['output_dir'],
            f"{self.config['experiment']['name']}_{timestamp}"
        )
        
        self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoints")
       
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        
        logger.info(f"Experiment directory: {self.exp_dir}")

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model and training state from checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle DataParallel wrapping
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_loss = checkpoint['best_loss']
            
            logger.info(f"Resumed training from checkpoint: {checkpoint_path}")
            logger.info(f"Resuming from epoch {self.start_epoch}, previous best loss: {self.best_loss:.4f}")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint {checkpoint_path}: {str(e)}")
            raise

    def _save_checkpoint(
        self,
        epoch: int,
        current_loss: float,
        is_best: bool = False,
        emergency: bool = False
    ) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict() if isinstance(self.model, nn.DataParallel)
                              else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        # Regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_epoch_{epoch}.pth"
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model separately
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with loss {current_loss:.4f}")
            
        # Emergency save has different naming
        if emergency:
            emergency_path = os.path.join(self.checkpoint_dir, f"emergency_epoch_{epoch}.pth")
            torch.save(checkpoint, emergency_path)
            logger.warning(f"Emergency checkpoint saved at epoch {epoch}")
        else:
            logger.info(f"Checkpoint saved at epoch {epoch}")


    @staticmethod
    def vae_loss(
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute VAE loss with beta weighting."""
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl_loss, recon_loss, kl_loss

    def train(self) -> Dict[str, list]:
        """Main training loop with checkpointing and visualization."""
        # Validation (optional)
        
        history = {
            'total_loss': [],
            'recon_loss': [],
            'kl_loss': [],
            'val_loss': []
        }
       
         # Create a unique history file for this training session
        history_file_path = os.path.join(self.exp_dir, "training_history.csv")
        iteration_history_file_path = os.path.join(self.exp_dir, "iteration_history.csv")
    
        with open(history_file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Write the header row
            writer.writerow(['epoch', 'total_loss', 'recon_loss', 'kl_loss', 'val_loss'])
        
          # Write iteration-level header
        with open(iteration_history_file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['epoch', 'iteration', 'total_loss', 'recon_loss', 'kl_loss'])
        
        
        try:
            for epoch in range(self.start_epoch, self.config['training']['epochs']):
                self.current_epoch = epoch
                self.model.train()
                epoch_loss = epoch_recon = epoch_kl = 0
                
                # Training phase
                with tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.config['training']['epochs']}") as pbar:
                    # for batch in pbar:
                    for iteration, batch in enumerate(pbar):
                        batch = batch.to(self.device)
                        # batch = batch.permute(0, 2, 1, 3, 4)  # Don't permute for difference dataset
                        
                        # Forward pass with mixed precision
                        with torch.amp.autocast('cuda',
                            enabled=self.config['training'].get('use_amp', False)
                        ):
                            recon_batch, mu, logvar = self.model(batch)
                            loss, recon_loss, kl_loss = self.vae_loss(
                                recon_batch,
                                batch,
                                mu,
                                logvar,
                                beta=self.config['training'].get('beta', 1.0)
                            )
                        
                        # Backward pass
                        self.optimizer.zero_grad()
                        self.scaler.scale(loss).backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['training'].get('grad_clip', 1.0)
                        )
                        
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        
                        # Update metrics
                        epoch_loss += loss.item()
                        epoch_recon += recon_loss.item()
                        epoch_kl += kl_loss.item()
                        with open(iteration_history_file_path, mode='a', newline='') as csv_file:
                            writer = csv.writer(csv_file)
                            writer.writerow([epoch + 1, iteration + 1, loss.item(), recon_loss.item(), kl_loss.item()])
                        
                       
                       
                            
                        pbar.set_postfix({
                            'loss': loss.item(),
                            'recon': recon_loss.item(),
                            'kl': kl_loss.item(),
                            # 'val_loss': val_loss if val_loss is not None else None
                        })
                # Calculate epoch metrics
                dataset_size = len(self.dataloader.dataset)
                total_loss = epoch_loss / dataset_size
                recon_loss = epoch_recon / dataset_size
                kl_loss = epoch_kl / dataset_size
                history['total_loss'].append(epoch_loss / dataset_size)
                history['recon_loss'].append(epoch_recon / dataset_size)
                history['kl_loss'].append(epoch_kl / dataset_size)
                
                # Validation (optional)
                val_loss = self._validate() if self.val_loader is not None else None
                if val_loss is not None:
                    history['val_loss'].append(val_loss)
              
                logger.info(f"Total loss per epoch: {total_loss}")
                
                
                  # Save epoch data to the history file
                with open(history_file_path, mode='a', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([epoch + 1, total_loss, recon_loss, kl_loss, val_loss if val_loss is not None else 'N/A'])
                
                
                
                
                # Checkpointing
                current_loss = val_loss if val_loss is not None else (epoch_loss / dataset_size)
                is_best = current_loss < self.best_loss
                if is_best:
                    self.best_loss = current_loss
                
                # Save checkpoint
                if (epoch + 1) % self.config['training'].get('checkpoint_interval', 1) == 0:
                    self._save_checkpoint(epoch, current_loss, is_best=is_best)
                
               
                
        except Exception as e:
            logger.error(f"Training failed at epoch {epoch}: {str(e)}")
            self._save_checkpoint(epoch, self.best_loss, emergency=True)
            raise
        
        # Save final model
        self._save_checkpoint(epoch, self.best_loss)
        logger.info("Training completed successfully")
        return history

    def _validate(self) -> float:
        """Run validation if validation loader is configured."""
        val_loader = self.val_loader
        if val_loader is None:
            return 0.0
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                # batch = batch.permute(0, 2, 1, 3, 4)  # Don't permute for difference dataset
                recon_batch, mu, logvar = self.model(batch)
                loss, _, _ = self.vae_loss(
                    recon_batch,
                    batch,
                    mu,
                    logvar,
                    beta=self.config['training'].get('beta', 1.0)
                )
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader.dataset)
        logger.info(f"Validation loss: {avg_loss:.4f}")
        return avg_loss

    def get_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent checkpoint in experiment directory."""
        try:
            checkpoints = []
            for f in os.listdir(self.checkpoint_dir):
                if f.startswith('checkpoint_epoch') and f.endswith('.pth'):
                    epoch = int(f.split('_')[2].split('.')[0])
                    checkpoints.append((epoch, f))
            
            if not checkpoints:
                return None
                
            checkpoints.sort()
            return os.path.join(self.checkpoint_dir, checkpoints[-1][1])
        except Exception as e:
            logger.error(f"Error finding checkpoints: {str(e)}")
            return None