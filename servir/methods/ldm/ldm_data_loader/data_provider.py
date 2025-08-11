import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from typing import Tuple, Optional, Dict
import math
import warnings
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import torch.nn as nn



    
# ######################## EXPLANATION####################################
'''The MultiModalForecastDataset class is a custom PyTorch dataset designed for
 multi-modal precipitation forecasting tasks, where both IMERG precipitation data
 and infrared (IR) brightness temperature data are used. It constructs fixed-size 
 rolling windows for supervised learning, where each training sample consists of an 
 input sequence and a corresponding future output sequence from the IMERG dataset, along
 with a parallel input sequence from the IR dataset. Specifically, the dataset extracts 
 12 time steps of IMERG data for both input and prediction, and 16 time steps of IR data 
 sampled at a given stride starting from a specified offset. This setup is especially useful
 when working with models that forecast future weather conditions by combining precipitation measurements
(from IMERG) and high-temporal-resolution cloud temperature data (from IR sensors).'''



class LatentMultiModalForecastDataset(Dataset):
    """
    PyTorch Dataset for multi-modal precipitation forecasting in latent space.
    Encodes IMERG and IR data using pre-trained VAEs before returning.
    """
    
    def __init__(self, 
                 imerg_file: str,
                 ir_file: str,
                 vae_imerg: Optional[nn.Module] = None,
                 vae_ir: Optional[nn.Module] = None,
                 imerg_steps: int = 12,
                 ir_steps: int = 16,
                 imerg_stride: int = 1,
                 ir_stride: int = 2,
                 ir_start_offset: int = 20,
                 imerg_precip_name: str = 'precipitations',
                 ir_tb_name: str = 'Tb',
                 device: str = 'cuda',
                 return_original: bool = False,
                 ):
        
        super().__init__()
        self.imerg_file = imerg_file
        self.ir_file = ir_file
        self.vae_imerg = vae_imerg
        self.vae_ir = vae_ir
        self.imerg_steps = imerg_steps
        self.ir_steps = ir_steps
        self.imerg_stride = imerg_stride
        self.ir_stride = ir_stride
        self.ir_start_offset = ir_start_offset
        self.imerg_precip_name = imerg_precip_name
        self.ir_tb_name = ir_tb_name
        self.device = device
        self.return_original = return_original
        self._first_encode = True  # Debug flag
        
        # Set VAEs to eval mode if provided
        if self.vae_imerg is not None:
            self.vae_imerg.eval()
        if self.vae_ir is not None:
            self.vae_ir.eval()
       
        
        # Open files to get metadata
        with h5py.File(self.imerg_file, 'r') as h5_imerg, h5py.File(self.ir_file, 'r') as h5_ir:
            # IMERG data info
            self.total_imerg = h5_imerg[imerg_precip_name].shape[0]
            self.height = h5_imerg[imerg_precip_name].shape[1]
            self.width = h5_imerg[imerg_precip_name].shape[2]
            
            # IR data info
            self.total_ir = h5_ir[ir_tb_name].shape[0]
            assert h5_ir[ir_tb_name].shape[1:] == (self.height, self.width)
            
            # Calculate maximum possible number of sequences based on IMERG
            max_sequences = math.floor(
                (self.total_imerg - (imerg_steps * 2)) / imerg_stride) + 1
            
            # Calculate required IR for each sequence and find the actual number we can use
            self.num_sequences = max_sequences
            for n in range(max_sequences, 0, -1):
                required_ir = (n - 1) * imerg_stride * 2 + ir_start_offset + ir_steps * ir_stride
                if self.total_ir >= required_ir:
                    self.num_sequences = n
                    break
            
            if self.num_sequences < max_sequences:
                warnings.warn(f"Reduced number of sequences from {max_sequences} to {self.num_sequences} due to insufficient IR data")
            
            if self.num_sequences == 0:
                raise ValueError("Not enough data to create even a single sequence")
    
    def encode_to_latent(self, data: torch.Tensor, vae: nn.Module, data_type: str = 'imerg') -> torch.Tensor:
        """
        Encode data to latent space using VAE.
        
        Args:
            data: Input tensor of shape (T, C, H, W)
            vae: VAE model to use for encoding
            data_type: Type of data ('imerg' or 'ir')
            
        Returns:
            Latent tensor
        """
        with torch.no_grad():
            # Input data shape: (T, C, H, W)
            # VAE expects: (batch, C, T, H, W)
            
            # Permute dimensions: (T, C, H, W) -> (C, T, H, W)
            data = data.permute(1, 0, 2, 3)
            
            # Add batch dimension and move to device
            data = data.unsqueeze(0).to(self.device)  # Now (1, C, T, H, W)
            
            # Print shape for debugging on first call
            # if hasattr(self, '_first_encode') and self._first_encode:
            #     print(f"Data shape going into VAE.encode: {data.shape}")
            #     self._first_encode = False
            
            # Encode to latent space
            mu, logvar = vae.encode(data)
            # Use mean for deterministic encoding during inference
            latent = mu
            
            # Remove batch dimension and permute back
            latent = latent.squeeze(0)  # (C_latent, T_latent, H_latent, W_latent)
            latent = latent.permute(1, 0, 2, 3)  # Back to (T_latent, C_latent, H_latent, W_latent)
            
        return latent.cpu()
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Calculate IMERG indices (30-min data)
        imerg_start = idx * self.imerg_stride
        imerg_mid = imerg_start + self.imerg_steps
        imerg_end = imerg_mid + self.imerg_steps
        
        # Calculate IR indices (15-min data)
        # Start from offset and roll by 2 images each sequence
        ir_base = self.ir_start_offset + (idx * 2)
        ir_indices = range(ir_base, ir_base + (self.ir_steps * self.ir_stride), self.ir_stride)
        
        # Read data
        with h5py.File(self.imerg_file, 'r') as h5_imerg, h5py.File(self.ir_file, 'r') as h5_ir:
            imerg_input = h5_imerg[self.imerg_precip_name][imerg_start:imerg_mid]
            imerg_output = h5_imerg[self.imerg_precip_name][imerg_mid:imerg_end]
            ir_input = h5_ir[self.ir_tb_name][ir_indices]
        
        # Convert to tensors and add channel dim
        imerg_input = torch.from_numpy(imerg_input).float().unsqueeze(1)
        imerg_output = torch.from_numpy(imerg_output).float().unsqueeze(1)
        ir_input = torch.from_numpy(ir_input).float().unsqueeze(1)
        
        # Create output dictionary
        output_dict = {}
        
        # Encode to latent space if VAEs are provided
        if self.vae_imerg is not None:
            # Debug print first time
            # if idx == 0:
                # print(f"IMERG input shape before encoding: {imerg_input.shape}")
                # print(f"Expected VAE input shape: (batch=1, channels=1, time=12, height=360, width=516)")
            imerg_input_latent = self.encode_to_latent(imerg_input, self.vae_imerg, 'imerg')
            imerg_output_latent = self.encode_to_latent(imerg_output, self.vae_imerg, 'imerg')
            # if idx == 0:
                # print(f"IMERG input latent shape after encoding: {imerg_input_latent.shape}")
            output_dict['imerg_input_latent'] = imerg_input_latent
            output_dict['imerg_output_latent'] = imerg_output_latent
        
        if self.vae_ir is not None:
            ir_input_latent = self.encode_to_latent(ir_input, self.vae_ir, 'ir')
            output_dict['ir_input_latent'] = ir_input_latent
        
        # Always return original data for visualization
        output_dict['imerg_input'] = imerg_input
        output_dict['imerg_output'] = imerg_output
        output_dict['ir_input'] = ir_input
        
        return output_dict


class MultiModalForecastDataset(Dataset):
    """
    PyTorch Dataset for multi-modal precipitation forecasting with:
    - 12-step IMERG input/output windows
    - 16-step IR input windows starting from 20th image with stride 1
    
    Args:
        imerg_file (str): Path to IMERG H5 file
        ir_file (str): Path to IR H5 file
        imerg_steps (int): IMERG input/output window size (default 12)
        ir_steps (int): IR input window size (default 16)
        imerg_stride (int): IMERG sequence stride (default 1)
        ir_stride (int): IR sampling stride within window (default 2)
        ir_start_offset (int): Starting offset for IR (default 20)
        imerg_precip_name (str): IMERG dataset name (default 'precipitations')
        ir_tb_name (str): IR dataset name (default 'Tb')
        transform (callable, optional): Optional transform for both data types
    """
    
    def __init__(self, 
                 imerg_file: str,
                 ir_file: str,
                 imerg_steps: int = 12,
                 ir_steps: int = 16,
                 imerg_stride: int = 1,
                 ir_stride: int = 2,
                 ir_start_offset: int = 20,
                 imerg_precip_name: str = 'precipitations',
                 ir_tb_name: str = 'Tb',
                 ):
        
        super().__init__()
        self.imerg_file = imerg_file
        self.ir_file = ir_file
        self.imerg_steps = imerg_steps
        self.ir_steps = ir_steps
        self.imerg_stride = imerg_stride
        self.ir_stride = ir_stride
        self.ir_start_offset = ir_start_offset
        self.imerg_precip_name = imerg_precip_name
        self.ir_tb_name = ir_tb_name
       
        
        # Open files to get metadata
        with h5py.File(self.imerg_file, 'r') as h5_imerg, h5py.File(self.ir_file, 'r') as h5_ir:
            # IMERG data info
            self.total_imerg = h5_imerg[imerg_precip_name].shape[0]
            self.height = h5_imerg[imerg_precip_name].shape[1]
            self.width = h5_imerg[imerg_precip_name].shape[2]
            
            # IR data info
            self.total_ir = h5_ir[ir_tb_name].shape[0]
            assert h5_ir[ir_tb_name].shape[1:] == (self.height, self.width)
            
            # Calculate maximum possible number of sequences based on IMERG
            max_sequences = math.floor(
                (self.total_imerg - (imerg_steps * 2)) / imerg_stride) + 1
            
            # Calculate required IR for each sequence and find the actual number we can use
            self.num_sequences = max_sequences
            for n in range(max_sequences, 0, -1):
                required_ir = (n - 1) * imerg_stride * 2 + ir_start_offset + ir_steps * ir_stride
                if self.total_ir >= required_ir:
                    self.num_sequences = n
                    break
            
            if self.num_sequences < max_sequences:
                warnings.warn(f"Reduced number of sequences from {max_sequences} to {self.num_sequences} due to insufficient IR data")
            
            if self.num_sequences == 0:
                raise ValueError("Not enough data to create even a single sequence")
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Calculate IMERG indices (30-min data)
        imerg_start = idx * self.imerg_stride
        imerg_mid = imerg_start + self.imerg_steps
        imerg_end = imerg_mid + self.imerg_steps
        
        # Calculate IR indices (15-min data)
        # Start from offset and roll by 2 images each sequence
        ir_base = self.ir_start_offset + (idx * 2)
        ir_indices = range(ir_base, ir_base + (self.ir_steps * self.ir_stride), self.ir_stride)
        
        # Read data
        with h5py.File(self.imerg_file, 'r') as h5_imerg, h5py.File(self.ir_file, 'r') as h5_ir:
            imerg_input = h5_imerg[self.imerg_precip_name][imerg_start:imerg_mid]
            imerg_output = h5_imerg[self.imerg_precip_name][imerg_mid:imerg_end]
            ir_input = h5_ir[self.ir_tb_name][ir_indices]
        
        # Convert to tensors and add channel dim
        imerg_input = torch.from_numpy(imerg_input).float().unsqueeze(1)
        imerg_output = torch.from_numpy(imerg_output).float().unsqueeze(1)
        ir_input = torch.from_numpy(ir_input).float().unsqueeze(1)
        
        
        
        return imerg_input, imerg_output, ir_input


def get_mm_dataloader(imerg_file: str,
                     ir_file: str,
                     batch_size: int = 4,
                     shuffle: bool = True,
                     num_workers: int = 4,
                     **kwargs) -> DataLoader:
    """Create DataLoader for multi-modal forecasting"""
    dataset = MultiModalForecastDataset(imerg_file, ir_file, **kwargs)
    
    def collate_fn(batch):
        imerg_in = torch.stack([item[0] for item in batch])
        imerg_out = torch.stack([item[1] for item in batch])
        ir_in = torch.stack([item[2] for item in batch])
        return imerg_in, imerg_out, ir_in
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )


def get_latent_mm_dataloader(imerg_file: str,
                            ir_file: str,
                            vae_imerg: Optional[nn.Module] = None,
                            vae_ir: Optional[nn.Module] = None,
                            batch_size: int = 4,
                            shuffle: bool = True,
                            num_workers: int = 0,  # Set to 0 for CUDA operations
                            device: str = 'cuda',
                            return_original: bool = False,
                            **kwargs) -> DataLoader:
    """
    Create DataLoader for multi-modal forecasting in latent space.
    
    Args:
        imerg_file: Path to IMERG H5 file
        ir_file: Path to IR H5 file
        vae_imerg: Pre-trained VAE for IMERG data
        vae_ir: Pre-trained VAE for IR data
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of workers (set to 0 for CUDA operations)
        device: Device to use for encoding
        return_original: Whether to return original data alongside latent
        **kwargs: Additional arguments for dataset
        
    Returns:
        DataLoader that returns dictionaries with latent representations
    """
    dataset = LatentMultiModalForecastDataset(
        imerg_file, 
        ir_file, 
        vae_imerg=vae_imerg,
        vae_ir=vae_ir,
        device=device,
        return_original=return_original,
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
    dataloader = get_mm_dataloader(
        imerg_file='/home1/ppatel2025/ldm_data_loader/imerg_data.h5',
        ir_file='/home1/ppatel2025/ldm_data_loader/filled_missing_nan_ir_data.h5',
        batch_size=8,
        imerg_steps=12,
        ir_steps=16,
        imerg_stride=1,
        ir_stride=1,
        ir_start_offset=20
    )
    
    for imerg_in, imerg_out, ir_in in dataloader:
        print(f"IMERG Input: {imerg_in.shape}")  # (8, 12, 1, 360, 560)
        print(f"IMERG Output: {imerg_out.shape}")  # (8, 12, 1, 360, 560)
        print(f"IR Input: {ir_in.shape}")  # (8, 16, 1, 360, 560)
        break
    
    