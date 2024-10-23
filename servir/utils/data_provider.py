import random
import h5py
import numpy as np
from torch.utils.data import Dataset
from numpy.lib.stride_tricks import sliding_window_view

import pytorch_lightning as L
from torch.utils.data.dataloader import DataLoader

class ImergDataset(Dataset):
    def __init__(self, imerg_filename, ir_filename, start_index, end_index, forecast_steps, history_steps, normalize_data=False):
        super(ImergDataset, self).__init__()
        self.imerg_filename = imerg_filename
        self.ir_filename = ir_filename
        self.normalize_data = normalize_data
        with h5py.File(self.imerg_filename, 'r') as hf:
            precipitation_time_series = hf['precipitations'][:].astype(np.float32)
            mean_imerg = hf['mean'][()]
            std_imerg = hf['std'][()]
            print("original shape", precipitation_time_series.shape)
            precipitation_time_series = precipitation_time_series[start_index*31*48:end_index*31*48]
            num_days_in_oct = 31
            num_years = end_index - start_index
            
            for i in range(num_years):
                monthly_precipitation_time_series = precipitation_time_series[i*num_days_in_oct*48: (i+1)*num_days_in_oct*48]


                monthly_input_precipitation = sliding_window_view(monthly_precipitation_time_series,
                                                        window_shape=history_steps, 
                                                        axis=0)
                if i == 0:
                    self.output_precipitation = sliding_window_view(monthly_precipitation_time_series[history_steps:],
                                                            window_shape=forecast_steps, 
                                                            axis=0)
                    self.input_precipitation = monthly_input_precipitation[:-forecast_steps]
                else:
                    self.output_precipitation = np.concatenate((self.output_precipitation, sliding_window_view(monthly_precipitation_time_series[history_steps:],
                                                            window_shape=forecast_steps, 
                                                            axis=0)))
                    self.input_precipitation = np.concatenate((self.input_precipitation, monthly_input_precipitation[:-forecast_steps]))

            
            # reshape to DGMR expected input
            self.input_precipitation = np.transpose(self.input_precipitation, (0, 3, 1, 2))
            if self.normalize_data == True:
                self.input_precipitation = (self.input_precipitation[:,:,None,:,:]-mean_imerg)/std_imerg
                self.input_precipitation = np.nan_to_num(self.input_precipitation, 0.)
            else:
                self.input_precipitation = self.input_precipitation[:,:,None,:,:]
            
            self.output_precipitation = np.transpose(self.output_precipitation, (0, 3, 1, 2))
            if self.normalize_data == True:
                self.output_precipitation = (self.output_precipitation[:,:,None,:,:]-mean_imerg)/std_imerg
                self.output_precipitation = np.nan_to_num(self.output_precipitation, 0.)
            else:
                 self.output_precipitation = self.output_precipitation[:,:,None,:,:]
            
            print("Precipitation Dataset input shape: ", self.input_precipitation.shape)
            print("Precipitation Dataset output shape: ", self.output_precipitation.shape)
        
        if self.ir_filename is not None:
            with h5py.File(self.ir_filename, 'r') as hf:
                IR_time_series = hf['IRs'][:].astype(np.float32)
                mean_ir = hf['mean'][()]
                std_ir = hf['std'][()]
                print("IR original shape", IR_time_series.shape)
                IR_time_series = IR_time_series[start_index*31*48*2:end_index*31*48*2]
                num_days_in_oct = 31
                num_years = end_index - start_index
                history_steps_IR =history_steps*2
                forecast_steps_IR = forecast_steps*2
                for i in range(num_years):
                    monthly_IR_time_series = IR_time_series[i*num_days_in_oct*48*2: (i+1)*num_days_in_oct*48*2]
                    monthly_input_IR= sliding_window_view(monthly_IR_time_series,
                                                            window_shape=history_steps_IR, 
                                                            axis=0)
                    if i == 0:
                        output_IR_sample = sliding_window_view(monthly_IR_time_series[history_steps_IR:],
                                                                window_shape=forecast_steps_IR, 
                                                                axis=0)[::2]
                        input_IR_sample = monthly_input_IR[:-forecast_steps_IR][::2]
                        
                        # move 9 images from output IR to input IR (since we have up to 1 15h of IR in the past)
                        self.input_IR = np.concatenate((input_IR_sample, output_IR_sample[:,:,:,0:9]), axis=3)
                        self.output_IR = output_IR_sample[:,:,:,9:]              
                    else:
                        
                        output_IR_sample = sliding_window_view(monthly_IR_time_series[history_steps_IR:],
                                                                window_shape=forecast_steps_IR, 
                                                                axis=0)[::2]
                        input_IR_sample = monthly_input_IR[:-forecast_steps_IR][::2]
                        # move 9 images from output IR to input IR (since we have up to 1 15h of IR in the past)
                        self.input_IR = np.concatenate((self.input_IR, np.concatenate((input_IR_sample, output_IR_sample[:,:,:,0:9]), axis=3)))
                        self.output_IR = np.concatenate((self.output_IR,output_IR_sample[:,:,:,9:]))
                        
            
                # reshape to DGMR expected input
                self.input_IR = np.transpose(self.input_IR, (0, 3, 1, 2))
                if self.normalize_data == True:
                    self.input_IR = (self.input_IR[:,-16:,None,:,:]-mean_ir)/std_ir
                    self.input_IR = np.nan_to_num(self.input_IR, 0.)
                else:
                    self.input_IR = self.input_IR[:,-16:,None,:,:]
                
                self.output_IR = np.transpose(self.output_IR, (0, 3, 1, 2))
                if self.normalize_data == True:
                    self.output_IR = (self.output_IR[:,:,None,:,:]-mean_ir)/std_ir
                    self.output_IR = np.nan_to_num(self.output_IR, 0.)
                else:
                    self.output_IR = self.output_IR[:,:,None,:,:]
                    
                print("IR Dataset input shape: ", self.input_IR.shape)
                print("IR Dataset output shape: ", self.output_IR.shape)
            
    # code obtained from https://stackoverflow.com/questions/19349410/how-to-pad-with-zeros-a-tensor-along-some-axis-python
    def symmetric_pad_array(self, input_array: np.ndarray, target_shape: tuple, pad_value: int) -> np.ndarray:
        for dim_in, dim_target in zip(input_array.shape, target_shape):
            if dim_target < dim_in:
                raise Exception("`target_shape` should be greater or equal than `input_array` shape for each axis.")

        pad_width = []
        for dim_in, dim_target  in zip(input_array.shape, target_shape):
            if (dim_in-dim_target)%2 == 0:
                pad_width.append((int(abs((dim_in-dim_target)/2)), int(abs((dim_in-dim_target)/2))))
            else:
                pad_width.append((int(abs((dim_in-dim_target)/2)), (int(abs((dim_in-dim_target)/2))+1)))
        
        return np.pad(input_array, pad_width, 'constant', constant_values=pad_value)
    
    def __getitem__(self, idx):
        return self.input_precipitation[idx], self.input_IR[idx], self.output_precipitation[idx]
    
    def __len__(self):
        return len(self.output_precipitation)

class IMERGDataModule(L.LightningDataModule):
    """
    Example of LightningDataModule for h5py IMERS dataset.
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        num_workers: int = 1,
        pin_memory: bool = True,
        forecast_steps = 12,
        history_steps = 12,
        imerg_filename = "/home1/aaravamudan2014/data/ghana_imerg_2011_2020_Oct.h5",
        ir_filename = "/home1/aaravamudan2014/data/wa_ir.h5",
        batch_size = 32
    ):
        """
        fake_data: random data is created and used instead. This is useful for testing
        """
        super().__init__()

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.dataloader_config = dict(
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            prefetch_factor=8,
            persistent_workers=True,
            # Disable automatic batching because dataset
            # returns complete batches.
            batch_size=None,
        )  
        
        self.train_dataset = ImergDataset(imerg_filename, 
                                ir_filename, 
                                0, 
                                8, 
                                forecast_steps, 
                                history_steps)

        self.val_dataset = ImergDataset(imerg_filename, 
                                ir_filename, 
                                8, 
                                9, 
                                forecast_steps, 
                                history_steps)
        self.test_dataset = ImergDataset(imerg_filename, 
                                ir_filename, 
                                9, 
                                10, 
                                forecast_steps, 
                                history_steps)
        
    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=2)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=2)
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2)
        return dataloader