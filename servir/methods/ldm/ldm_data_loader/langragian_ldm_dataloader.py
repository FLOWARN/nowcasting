import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from typing import Tuple, Optional, Dict
import math
import warnings
import sys
import os

# Add the local residual_vae path to import langragian and vae
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'residual_vae', 'src'))
from extrapolation_methods.langragian_persistance import langragian_persistance
from models.vae import SimpleVAE3D

class LangragianLDMDataset(Dataset):
    """
    Dataset for LDM with encoded(gt-langragian) + encoded(ir) as context.
    
    Data structure:
    - First Langragian call: GT[0:12] → LP[13:24] (12 timesteps)
    - Second Langragian call: GT[13:24] → LP[25:36] (12 timesteps)
    - Input context: encode[GT[13:24] - LP[13:24]] + encode[IR[44:60]]
    - Output label: encode[GT[25:36] - LP[25:36]]
    
    Args:
        imerg_file: Path to IMERG H5 file
        ir_file: Path to IR H5 file  
        vae_checkpoint_path: Path to pre-trained VAE model
        ir_start_offset: Starting offset for IR data (default 44, which is 11th hour)
        ir_steps: Number of IR timesteps to use (default 16)
        ir_stride: IR sampling stride (default 1 for consecutive timesteps)
        imerg_precip_name: IMERG dataset name (default 'precipitations')
        ir_tb_name: IR dataset name (default 'IRs')
        device: Device for VAE encoding
        normalize_data: Whether to normalize data
    """
    
    def __init__(self, 
                 imerg_file: str,
                 ir_file: str,
                 vae_checkpoint_path: str,
                 ir_start_offset: int = 44,
                 ir_steps: int = 16,
                 ir_stride: int = 1,
                 imerg_precip_name: str = 'precipitations',
                 ir_tb_name: str = 'Tb',
                 device: str = 'cuda',
                 normalize_data: bool = False):
        
        super().__init__()
        self.imerg_file = imerg_file
        self.ir_file = ir_file
        self.vae_checkpoint_path = vae_checkpoint_path
        self.ir_start_offset = ir_start_offset
        self.ir_steps = ir_steps
        self.ir_stride = ir_stride
        self.imerg_precip_name = imerg_precip_name
        self.ir_tb_name = ir_tb_name
        self.device = device
        self.normalize_data = normalize_data
        
        # Load VAE model for encoding difference data
        self.vae_model = SimpleVAE3D(input_channels=1, latent_dim=32)
        if os.path.exists(vae_checkpoint_path):
            checkpoint = torch.load(vae_checkpoint_path, map_location=device, weights_only=False)
            self.vae_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded difference VAE model from {vae_checkpoint_path}")
        else:
            print(f"Warning: Difference VAE checkpoint not found at {vae_checkpoint_path}")
        
        self.vae_model.to(device)
        self.vae_model.eval()
        
        # Load separate VAE model for encoding IR data
        self.ir_vae_model = SimpleVAE3D(input_channels=1, latent_dim=32)
        ir_checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'vae_models', 'ir_model_final.pth')
        if os.path.exists(ir_checkpoint_path):
            ir_checkpoint = torch.load(ir_checkpoint_path, map_location=device, weights_only=False)
            self.ir_vae_model.load_state_dict(ir_checkpoint['model_state_dict'])
            
            # Load IR model's normalization stats
            self.ir_mean = ir_checkpoint.get('mean', 0.0)
            self.ir_std = ir_checkpoint.get('std', 1.0)
            
            # Handle cases where stats might be tensors or scalars
            if isinstance(self.ir_mean, torch.Tensor):
                self.ir_mean = self.ir_mean.item()
            if isinstance(self.ir_std, torch.Tensor):
                self.ir_std = self.ir_std.item()
                
            print(f"Loaded IR VAE model from {ir_checkpoint_path}")
            print(f"IR model normalization - mean: {self.ir_mean}, std: {self.ir_std}")
        else:
            print(f"Warning: IR VAE checkpoint not found at {ir_checkpoint_path}")
            self.ir_mean = 0.0
            self.ir_std = 1.0
        
        self.ir_vae_model.to(device)
        self.ir_vae_model.eval()
        
        # Calculate number of sequences similar to data_provider.py
        self._calculate_sequences()
    
    def _calculate_sequences(self):
        """Calculate number of valid sequences following data_provider.py pattern."""
        # Open files to get metadata
        with h5py.File(self.imerg_file, 'r') as h5_imerg, h5py.File(self.ir_file, 'r') as h5_ir:
            # IMERG data info
            self.total_imerg = h5_imerg[self.imerg_precip_name].shape[0]
            self.height = h5_imerg[self.imerg_precip_name].shape[1]
            self.width = h5_imerg[self.imerg_precip_name].shape[2]
            
            # IR data info  
            self.total_ir = h5_ir[self.ir_tb_name].shape[0]
            assert h5_ir[self.ir_tb_name].shape[1:] == (self.height, self.width)
            
            # Load normalization stats
            self.mean_imerg = h5_imerg.get('mean', None)
            self.std_imerg = h5_imerg.get('std', None)
            self.mean_ir = h5_ir.get('mean', None)
            self.std_ir = h5_ir.get('std', None)
            
            # Handle cases where stats might not exist or be in different format
            if self.mean_imerg is not None:
                try:
                    self.mean_imerg = self.mean_imerg[()]
                except:
                    self.mean_imerg = 0.0
            else:
                self.mean_imerg = 0.0
                
            if self.std_imerg is not None:
                try:
                    self.std_imerg = self.std_imerg[()]
                except:
                    self.std_imerg = 1.0
            else:
                self.std_imerg = 1.0
                
            if self.mean_ir is not None:
                try:
                    self.mean_ir = self.mean_ir[()]
                except:
                    self.mean_ir = 0.0
            else:
                self.mean_ir = 0.0
                
            if self.std_ir is not None:
                try:
                    self.std_ir = self.std_ir[()]
                except:
                    self.std_ir = 1.0
            else:
                self.std_ir = 1.0
            
            # Calculate maximum sequences based on IMERG (need 36 timesteps)
            max_sequences = self.total_imerg - 36 + 1
            
            # Calculate required IR for each sequence and find actual usable sequences
            self.num_sequences = max_sequences
            for n in range(max_sequences, 0, -1):
                # IR calculation: ir_start_offset + (sequence_idx * 2) + ir_steps * ir_stride
                # This matches the logic in __getitem__: ir_base = self.ir_start_offset + (idx * 2)
                required_ir = self.ir_start_offset + ((n - 1) * 2) + self.ir_steps * self.ir_stride
                if self.total_ir >= required_ir:
                    self.num_sequences = n
                    break
            
            if self.num_sequences < max_sequences:
                warnings.warn(f"Reduced sequences from {max_sequences} to {self.num_sequences} due to insufficient IR data")
            
            if self.num_sequences == 0:
                raise ValueError("Not enough data to create even a single sequence")
            
            print(f"Dataset initialized with {self.num_sequences} sequences")
            print(f"IMERG shape: {(self.total_imerg, self.height, self.width)}")
            print(f"IR shape: {(self.total_ir, self.height, self.width)}")
    
    def encode_to_latent(self, data: torch.Tensor) -> torch.Tensor:
        """
        Encode difference data to latent space using difference VAE.
        
        Args:
            data: Input tensor of shape (T, C, H, W)
            
        Returns:
            Latent tensor
        """
        with torch.no_grad():
            # Permute dimensions: (T, C, H, W) -> (C, T, H, W)
            data = data.permute(1, 0, 2, 3)
            
            # Add batch dimension and move to device
            data = data.unsqueeze(0).to(self.device)  # Now (1, C, T, H, W)
            
            # Encode to latent space
            mu, logvar = self.vae_model.encode(data)
            # Use mean for deterministic encoding
            latent = mu
            
            # Remove batch dimension and permute back
            latent = latent.squeeze(0)  # (C_latent, T_latent, H_latent, W_latent)
            latent = latent.permute(1, 0, 2, 3)  # Back to (T_latent, C_latent, H_latent, W_latent)
            
        return latent.cpu()
    
    def encode_to_latent_ir(self, data: torch.Tensor) -> torch.Tensor:
        """
        Encode IR data to latent space using IR VAE with IR model's normalization.
        
        Args:
            data: Input tensor of shape (T, C, H, W)
            
        Returns:
            Latent tensor
        """
        with torch.no_grad():
            # Normalize using IR model's stats
            data = (data - self.ir_mean) / self.ir_std
            data = torch.nan_to_num(data, 0.0)
            
            # Permute dimensions: (T, C, H, W) -> (C, T, H, W)
            data = data.permute(1, 0, 2, 3)
            
            # Add batch dimension and move to device
            data = data.unsqueeze(0).to(self.device)  # Now (1, C, T, H, W)
            
            # Encode to latent space
            mu, logvar = self.ir_vae_model.encode(data)
            # Use mean for deterministic encoding
            latent = mu
            
            # Remove batch dimension and permute back
            latent = latent.squeeze(0)  # (C_latent, T_latent, H_latent, W_latent)
            latent = latent.permute(1, 0, 2, 3)  # Back to (T_latent, C_latent, H_latent, W_latent)
            
        return latent.cpu()
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample following the Langragian LDM structure.
        
        Returns:
            Dictionary with 'context' and 'target' keys containing encoded data
        """
        # Read IMERG data for this sequence (36 timesteps)
        with h5py.File(self.imerg_file, 'r') as h5_imerg:
            precip_sequence = h5_imerg[self.imerg_precip_name][idx:idx+36]
        
        # Split into segments
        gt_0_12 = precip_sequence[:12]          # GT[0:12] for first Langragian input
        gt_13_24 = precip_sequence[12:24]       # GT[13:24] for context and second Langragian input
        gt_25_36 = precip_sequence[24:36]       # GT[25:36] for target
        
        # Generate Langragian predictions in two steps
        # First call: Use GT[0:12] to predict next 12 timesteps [13:24]
        langragian_pred_13_24 = langragian_persistance(gt_0_12, timesteps=12)
        
        # Second call: Use GT[13:24] to predict next 12 timesteps [25:36]
        langragian_pred_25_36 = langragian_persistance(gt_13_24, timesteps=12)
        
        # Calculate differences
        diff_context = gt_13_24 - langragian_pred_13_24    # GT[13:24] - LP[13:24]
        diff_target = gt_25_36 - langragian_pred_25_36     # GT[25:36] - LP[25:36]
        
        # Read IR data with moving window (IR moves 2x faster due to 2x temporal resolution)
        ir_base = self.ir_start_offset + (idx * 2)
        ir_indices = range(ir_base, ir_base + (self.ir_steps * self.ir_stride), self.ir_stride)
        
        # Check bounds to prevent index error
        max_ir_index = max(ir_indices)
        with h5py.File(self.ir_file, 'r') as h5_ir:
            total_ir_available = h5_ir[self.ir_tb_name].shape[0]
            if max_ir_index >= total_ir_available:
                raise IndexError(f"IR index {max_ir_index} out of range for dataset with {total_ir_available} timesteps. "
                               f"Sample idx={idx}, ir_base={ir_base}, ir_indices={list(ir_indices)}")
            ir_context = h5_ir[self.ir_tb_name][ir_indices]
        
        # Convert to tensors and add channel dimension
        diff_context_tensor = torch.from_numpy(diff_context).float().unsqueeze(1)  # (12, 1, H, W)
        diff_target_tensor = torch.from_numpy(diff_target).float().unsqueeze(1)    # (12, 1, H, W)
        ir_context_tensor = torch.from_numpy(ir_context).float().unsqueeze(1)      # (16, 1, H, W)
        
        # Normalize difference data if required (IR data normalized in encode_to_latent_ir)
        if self.normalize_data:
            diff_context_tensor = (diff_context_tensor - self.mean_imerg) / self.std_imerg
            diff_context_tensor = torch.nan_to_num(diff_context_tensor, 0.0)
            
            diff_target_tensor = (diff_target_tensor - self.mean_imerg) / self.std_imerg
            diff_target_tensor = torch.nan_to_num(diff_target_tensor, 0.0)
        
        # Encode to latent space using respective VAE models
        encoded_diff_context = self.encode_to_latent(diff_context_tensor)
        encoded_diff_target = self.encode_to_latent(diff_target_tensor)
        encoded_ir_context = self.encode_to_latent_ir(ir_context_tensor)
        
        # Combine context: concatenate along channel dimension
        combined_context = torch.cat([encoded_diff_context, encoded_ir_context], dim=1)
        
        # Convert original data to tensors for output
        gt_context_tensor = torch.from_numpy(gt_13_24).float()    # GT[13:24]
        gt_target_tensor = torch.from_numpy(gt_25_36).float()     # GT[25:36]
        langragian_context_tensor = torch.from_numpy(langragian_pred_13_24).float()  # LP[13:24]
        langragian_target_tensor = torch.from_numpy(langragian_pred_25_36).float()   # LP[25:36]
        ir_raw_tensor = torch.from_numpy(ir_context).float()     # Raw IR data
        
        return {
            # Encoded data for training
            'context': combined_context,
            'target': encoded_diff_target,
            
            # Original data for plotting/visualization
            'gt_context': gt_context_tensor,           # GT[13:24] - context ground truth
            'gt_target': gt_target_tensor,             # GT[25:36] - target ground truth
            'langragian_context': langragian_context_tensor,  # LP[13:24] - context langragian
            'langragian_target': langragian_target_tensor,    # LP[25:36] - target langragian
            'ir_raw': ir_raw_tensor,                   # Raw IR data
            'diff_context': diff_context_tensor,       # GT[13:24] - LP[13:24] (normalized)
            'diff_target': diff_target_tensor          # GT[25:36] - LP[25:36] (normalized)
        }

def get_langragian_ldm_dataloader(imerg_file: str,
                                 ir_file: str,
                                 vae_checkpoint_path: str,
                                 batch_size: int = 4,
                                 shuffle: bool = True,
                                 num_workers: int = 0,  # Set to 0 for CUDA operations
                                 device: str = 'cuda',
                                 **kwargs) -> DataLoader:
    """
    Create DataLoader for Langragian LDM dataset following data_provider.py pattern.
    
    Args:
        imerg_file: Path to IMERG H5 file
        ir_file: Path to IR H5 file
        vae_checkpoint_path: Path to pre-trained VAE model
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of workers (set to 0 for CUDA operations)
        device: Device to use for encoding
        **kwargs: Additional arguments for dataset
        
    Returns:
        DataLoader that returns dictionaries with encoded data and original data
    """
    dataset = LangragianLDMDataset(
        imerg_file=imerg_file,
        ir_file=ir_file,
        vae_checkpoint_path=vae_checkpoint_path,
        device=device,
        **kwargs
    )
    
    def collate_fn(batch):
        # Initialize output dictionary
        collated = {}
        
        # Get all keys from first item
        keys = batch[0].keys()
        
        # Stack each key across batch
        for key in keys:
            collated[key] = torch.stack([item[key] for item in batch])
        
        return collated
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

# Example usage
if __name__ == "__main__":
    # Example paths - adjust according to your setup
    imerg_file = "imerg_data.h5"
    ir_file = "filled_missing_nan_ir_data.h5"
    vae_checkpoint = "../vae_models/diff_encoder.pth"
    
    # Create dataloader
    dataloader = get_langragian_ldm_dataloader(
        imerg_file=imerg_file,
        ir_file=ir_file,
        vae_checkpoint_path=vae_checkpoint,
        batch_size=8,
        ir_start_offset=44,
        ir_steps=16,
        ir_stride=1
    )
    
    print("Testing dataloader...")
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Context shape: {batch['context'].shape}")
        print(f"  Target shape: {batch['target'].shape}")
        if batch_idx >= 2:  # Test first 3 batches
            break
    
    print("Dataloader test completed!")