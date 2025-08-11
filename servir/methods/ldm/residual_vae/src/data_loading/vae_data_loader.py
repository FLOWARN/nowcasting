import random
import h5py
import numpy as np
from torch.utils.data import Dataset
import pytorch_lightning as L
from torch.utils.data.dataloader import DataLoader


class VAESequenceDataset(Dataset):
    def __init__(self, precipitation_time_series, sequence_length, mean_imerg=None, std_imerg=None, 
                 normalize_data=False, image_shape=(64,64)):
        super(VAESequenceDataset, self).__init__()
        self.precipitation_time_series = precipitation_time_series
        self.sequence_length = sequence_length
        self.mean_imerg = mean_imerg
        self.std_imerg = std_imerg
        self.normalize_data = normalize_data
        self.img_height = image_shape[0]
        self.img_width = image_shape[1]

        # Use original size without cropping
        self.img_height = self.precipitation_time_series.shape[1]
        self.img_width = self.precipitation_time_series.shape[2]

        print("Original shape:", precipitation_time_series.shape)
        
        # Create non-overlapping sequences
        total_timesteps = self.precipitation_time_series.shape[0]
        num_complete_sequences = total_timesteps // sequence_length
        
        # Truncate to ensure we have complete sequences only
        truncated_length = num_complete_sequences * sequence_length
        self.precipitation_time_series = self.precipitation_time_series[:truncated_length]
        
        # Reshape into non-overlapping sequences
        self.sequences = self.precipitation_time_series.reshape(
            num_complete_sequences, sequence_length, self.img_height, self.img_width
        )
        
        # Transpose to match expected format: (batch, time, height, width)
        self.sequences = np.transpose(self.sequences, (0, 1, 2, 3))
        
        # Add channel dimension and normalize if needed
        if self.normalize_data and self.mean_imerg is not None and self.std_imerg is not None:
            self.sequences = (self.sequences[:, None, :, :, :] - self.mean_imerg) / self.std_imerg
            self.sequences = np.nan_to_num(self.sequences, 0.)
        else:
            self.sequences = self.sequences[:, None, :, :, :]
        
        print(f"VAE Dataset shape: {self.sequences.shape}")
        print(f"Number of sequences: {len(self.sequences)}")
        print(f"Sequence length: {sequence_length}")
        
    def __getitem__(self, idx):
        return self.sequences[idx]
    
    def __len__(self):
        return len(self.sequences)


class VAEDataModule(L.LightningDataModule):
    def __init__(
        self,
        num_workers: int = 1,
        pin_memory: bool = True,
        sequence_length: int = 12,
        imerg_filename: str = "data/imerg_data.h5",
        batch_size: int = 10,
        image_shape: tuple = (64, 64),
        normalize_data: bool = False,
        train_split: float = 0.8,
        val_split: float = 0.1
    ):
        super().__init__()
        
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.imerg_filename = imerg_filename
        self.sequence_length = sequence_length
        self.image_shape = image_shape
        self.normalize_data = normalize_data
        self.train_split = train_split
        self.val_split = val_split
        
        # Load data
        with h5py.File(self.imerg_filename, 'r') as hf:
            precipitation_time_series = hf['precipitations'][:].astype(np.float32)
            
            if normalize_data:
                mean_imerg = hf['mean'][()]
                std_imerg = hf['std'][()]
            else:
                mean_imerg = None
                std_imerg = None
        
        # Create full dataset
        full_dataset = VAESequenceDataset(
            precipitation_time_series=precipitation_time_series,
            sequence_length=sequence_length,
            mean_imerg=mean_imerg,
            std_imerg=std_imerg,
            normalize_data=normalize_data,
            image_shape=image_shape
        )
        
        # Split dataset
        total_sequences = len(full_dataset)
        train_size = int(total_sequences * train_split)
        val_size = int(total_sequences * val_split)
        test_size = total_sequences - train_size - val_size
        
        # Create train/val/test splits
        self.train_sequences = full_dataset.sequences[:train_size]
        self.val_sequences = full_dataset.sequences[train_size:train_size+val_size]
        self.test_sequences = full_dataset.sequences[train_size+val_size:]
        
        print(f"Train sequences: {len(self.train_sequences)}")
        print(f"Val sequences: {len(self.val_sequences)}")
        print(f"Test sequences: {len(self.test_sequences)}")
        
    def train_dataloader(self):
        train_dataset = SimpleSequenceDataset(self.train_sequences)
        return DataLoader(train_dataset, batch_size=self.batch_size, 
                         num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=True)
    
    def val_dataloader(self):
        val_dataset = SimpleSequenceDataset(self.val_sequences)
        return DataLoader(val_dataset, batch_size=self.batch_size, 
                         num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=False)
    
    def test_dataloader(self):
        test_dataset = SimpleSequenceDataset(self.test_sequences)
        return DataLoader(test_dataset, batch_size=self.batch_size, 
                         num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=False)


class SimpleSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __getitem__(self, idx):
        return self.sequences[idx]
    
    def __len__(self):
        return len(self.sequences)