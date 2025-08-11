import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from typing import Tuple
import math
import warnings
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os



    
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
    
    
################# To visualize the data and create GIFs, uncomment the following section #################    
    
    # for imerg_in, imerg_out, ir_in in dataloader:
    #     print(f"IMERG Input: {imerg_in.shape}")  # (8, 12, 1, 360, 560)
    #     print(f"IMERG Output: {imerg_out.shape}")  # (8, 12, 1, 360, 560)
    #     print(f"IR Input: {ir_in.shape}")  # (8, 16, 1, 360, 560)
        
    #     # Take the first sample from the batch
    #     sample_idx = 0
    #     imerg_in_sample = imerg_in[sample_idx].squeeze().numpy()  # (12, 360, 560)
    #     imerg_out_sample = imerg_out[sample_idx].squeeze().numpy()  # (12, 360, 560)
    #     ir_in_sample = ir_in[sample_idx].squeeze().numpy()  # (16, 360, 560)
        
    #     # Create directory if it doesn't exist
    #     os.makedirs('sample_gifs', exist_ok=True)
        
    #     # Function to create proper temporal animation
    #     def create_temporal_gif(data, filename, title, cmap='viridis', vmin=None, vmax=None):
    #         fig, ax = plt.subplots(figsize=(10, 8))
            
    #         # Initialize plot with first frame
    #         img = ax.imshow(data[0], cmap=cmap, vmin=vmin, vmax=vmax)
    #         plt.colorbar(img, ax=ax)
    #         ax.set_title(f"{title} - Frame 1/{len(data)}")
            
    #         def update(frame):
    #             img.set_array(data[frame])
    #             ax.set_title(f"{title} - Frame {frame+1}/{len(data)}")
    #             return img,
            
    #         ani = animation.FuncAnimation(
    #             fig, 
    #             update, 
    #             frames=len(data), 
    #             interval=200,  # 200ms between frames
    #             blit=True
    #         )
            
    #         ani.save(f'sample_gifs/{filename}.gif', writer='pillow', fps=5)  # 5 frames per second
    #         plt.close()
        
    #     # Get consistent color scaling for precipitation
    #     precip_max = max(imerg_in_sample.max(), imerg_out_sample.max())
        
    #     # Create the GIFs
    #     create_temporal_gif(imerg_in_sample, 'imerg_input', 'IMERG Input', 
    #                        cmap='YlGnBu', vmin=0, vmax=precip_max)
    #     create_temporal_gif(imerg_out_sample, 'imerg_output', 'IMERG Output', 
    #                        cmap='YlGnBu', vmin=0, vmax=precip_max)
    #     create_temporal_gif(ir_in_sample, 'ir_input', 'IR Input', 
    #                        cmap='inferno')
        
    #     print("Saved temporal animation GIFs in 'sample_gifs' directory")
    #     break