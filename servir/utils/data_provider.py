import random
import h5py
import numpy as np
from torch.utils.data import Dataset
from numpy.lib.stride_tricks import sliding_window_view

import pytorch_lightning as L
from torch.utils.data.dataloader import DataLoader
import torch

class IMERGDataModuleLatentDim(Dataset):
    """
    Data module for IMERG data. This class is used to load the IMERG data and prepare it for training.
    It inherits from the Dataset class and implements the __getitem__ and __len__ methods.
    The __getitem__ method returns the input and output data for a given index.
    The __len__ method returns the length of the dataset.
    The data is loaded from the IMERG file and the IR file if provided.

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, imerg_filename, ir_filename, forecast_steps, history_steps, normalize_data=False,
                 batch_size = 32,
                 image_shape=(64, 64),
                 production_mode=False, mode=None):
        super(IMERGDataModuleLatentDim, self).__init__()
        self.imerg_filename = imerg_filename
        self.ir_filename = ir_filename
        self.normalize_data = normalize_data
        self.img_height = image_shape[0]
        self.img_width = image_shape[1]
        self.production_mode = production_mode
        self.forecast_steps = forecast_steps
        self.history_steps = history_steps
        self.batch_size = batch_size
        self.mode = mode
        if self.ir_filename is not None:
            self.user_IR = True
        else:
            self.user_IR = False

        num_samples = 281

        if mode == 0:
            num_samples = 281
        elif mode == 1:
            num_samples = 281
        elif mode == 2:
            num_samples = 281

        self.input_precipitation = torch.rand(num_samples, 32, 4,  64, 64)
        self.output_precipitation = torch.rand(num_samples, 32, 4, 64, 64)
        self.input_IR = torch.rand(num_samples,  32, 4, 64, 64)

        self.condition = torch.concat((self.input_precipitation, self.input_IR), 1 )



    def __getitem__(self, idx):
        return self.condition[idx] , self.output_precipitation[idx]

    def __len__(self):
        return len(self.input_precipitation)

    def get_input_timestamps(self):
        return self.input_timestamps






class ImergWADataset(Dataset):
    def __init__(self, imerg_filename, ir_filename, forecast_steps, history_steps, normalize_data=False,
                 image_shape=(64, 64),
                 production_mode=False, mode=None):
        """
        Dataset for IMERG data. This class is used to load the IMERG data and prepare it for training.
        It inherits from the Dataset class and implements the __getitem__ and __len__ methods.
        The __getitem__ method returns the input and output data for a given index.
        The __len__ method returns the length of the dataset.
        The data is loaded from the IMERG file and the IR file if provided.
        Args:
            imerg_filename (str): path to the IMERG file
            ir_filename (str): path to the IR file
            forecast_steps (int): number of forecast steps
            history_steps (int): number of history steps
            normalize_data (bool): whether to normalize the data
            image_shape (tuple): shape of the image
            production_mode (bool): whether to use production mode (i.e., there is no need to have output sequences (since it does not exist))
            mode (int): mode of the dataset, 0 for training, 1 for validation, 2 for testing
       
        """
        super(ImergWADataset, self).__init__()
        self.imerg_filename = imerg_filename
        self.ir_filename = ir_filename
        self.normalize_data = normalize_data
        self.img_height = image_shape[0]
        self.img_width = image_shape[1]
        self.production_mode = production_mode
        self.mode = mode
        if self.ir_filename is not None:
            self.user_IR = True
        else:
            self.user_IR = False

        # imerg_correction = 2
        with h5py.File(self.imerg_filename, 'r') as hf:
            precipitation_time_series = hf['precipitations'][:].astype(np.float32)
            timestamps = hf['timestamps'][:]

            # crop the image to the desired shape(center crop)
            # if self.img_height != precipitation_time_series.shape[1]:
            #     h_start = (precipitation_time_series.shape[1] - self.img_height) // 2
            #     precipitation_time_series = precipitation_time_series[:, h_start:h_start+self.img_height, :]

            # if self.img_width != precipitation_time_series.shape[2]:
            #     w_start = (precipitation_time_series.shape[2] - self.img_width) // 2
            #     precipitation_time_series = precipitation_time_series[:, :, w_start:w_start+self.img_width]
            mean_imerg = hf['mean'][()]
            std_imerg = hf['std'][()]
            print("original shape", precipitation_time_series.shape)

            self.input_precipitation = sliding_window_view(precipitation_time_series,
                                                           window_shape=history_steps,
                                                           axis=0)[:-forecast_steps]

            self.input_timestamps = sliding_window_view(timestamps,
                                                        window_shape=history_steps,
                                                        axis=0)[:-forecast_steps]

            if not production_mode:
                self.input_precipitation = sliding_window_view(precipitation_time_series,
                                                               window_shape=history_steps,
                                                               axis=0)[:-forecast_steps]

                self.input_timestamps = sliding_window_view(timestamps,
                                                            window_shape=history_steps,
                                                            axis=0)[:-forecast_steps]

                self.output_precipitation = sliding_window_view(precipitation_time_series[history_steps:],
                                                                window_shape=forecast_steps,
                                                                axis=0)
                self.output_timestamps = sliding_window_view(timestamps[history_steps:],
                                                             window_shape=forecast_steps,
                                                             axis=0)
            else:
                self.input_precipitation = sliding_window_view(precipitation_time_series,
                                                               window_shape=history_steps,
                                                               axis=0)

                self.input_timestamps = sliding_window_view(timestamps,
                                                            window_shape=history_steps,
                                                            axis=0)

            # reshape to DGMR expected input
            self.input_precipitation = np.transpose(self.input_precipitation, (0, 3, 1, 2))
            if self.normalize_data == True:
                self.input_precipitation = (self.input_precipitation[:, :, None, :, :] - mean_imerg) / std_imerg
                self.input_precipitation = np.nan_to_num(self.input_precipitation, 0.)
            else:
                self.input_precipitation = self.input_precipitation[:, :, None, :, :]

            if not production_mode:
                self.output_precipitation = np.transpose(self.output_precipitation, (0, 3, 1, 2))
                if self.normalize_data == True:
                    self.output_precipitation = (self.output_precipitation[:, :, None, :, :] - mean_imerg) / std_imerg
                    self.output_precipitation = np.nan_to_num(self.output_precipitation, 0.)
                else:
                    self.output_precipitation = self.output_precipitation[:, :, None, :, :]

        if self.user_IR:
            ir_correction = 1
            with h5py.File(self.ir_filename, 'r') as hf:
                IR_time_series = hf['precipitations'][:].astype(np.float32)[1:-ir_correction]
                datetime_series = hf['timestamps'][:][1:-ir_correction]

                # mean_ir = hf['mean'][()]
                # std_ir = hf['std'][()]

                if self.img_height != IR_time_series.shape[1]:
                    h_start = (IR_time_series.shape[1] - self.img_height) // 2
                    IR_time_series = IR_time_series[:, h_start:h_start + self.img_height, :]

                if self.img_width != IR_time_series.shape[2]:
                    w_start = (IR_time_series.shape[2] - self.img_width) // 2
                    IR_time_series = IR_time_series[:, :, w_start:w_start + self.img_width]

                # print("IR original shape", IR_time_series.shape)
                history_steps_IR = history_steps * 2
                forecast_steps_IR = forecast_steps * 2

                input_IR_sample = sliding_window_view(IR_time_series,
                                                      window_shape=history_steps_IR,
                                                      axis=0)

                input_IR_timestamps_sample = sliding_window_view(datetime_series,
                                                                 window_shape=history_steps_IR,
                                                                 axis=0)

                input_IR_sample = input_IR_sample[:-forecast_steps_IR][::2]
                input_IR_timestamps_sample = input_IR_timestamps_sample[:-forecast_steps_IR][::2]

                output_IR_sample = sliding_window_view(IR_time_series[history_steps_IR:],
                                                       window_shape=forecast_steps_IR,
                                                       axis=0)[::2]
                output_IR_timestamps_sample = sliding_window_view(datetime_series[history_steps_IR:],
                                                                  window_shape=forecast_steps_IR,
                                                                  axis=0)[::2]

                # move 9 images from output IR to input IR (since we have up to 1 15h of IR in the past)
                # print(input_IR_sample.shape)
                # print(output_IR_sample.shape)

                self.input_IR = np.concatenate((input_IR_sample, output_IR_sample[:, :, :, 0:9]), axis=3)
                self.input_IR_timestamps_sample = np.concatenate(
                    (input_IR_timestamps_sample, output_IR_timestamps_sample[:, 0:9]), axis=1)

                self.output_IR = output_IR_sample[:, :, :, 9:]
                self.output_IR_timestamps_sample = output_IR_timestamps_sample[:, 9:]

                # reshape to DGMR expected input
                self.input_IR = np.transpose(self.input_IR, (0, 3, 1, 2))
                if self.normalize_data == True:
                    self.input_IR = (self.input_IR[:, -16:, None, :, :] - mean_ir) / std_ir
                    self.input_IR = np.nan_to_num(self.input_IR, 0.)
                else:
                    self.input_IR = self.input_IR[:, -16:, None, :, :]

                self.output_IR = np.transpose(self.output_IR, (0, 3, 1, 2))

                # normalize data and add an extra channel to incorporate neural network expected input
                if self.normalize_data == True:
                    self.output_IR = (self.output_IR[:, :, None, :, :] - mean_ir) / std_ir
                    self.output_IR = np.nan_to_num(self.output_IR, 0.)
                else:
                    self.output_IR = self.output_IR[:, :, None, :, :]

                if self.mode is not None:
                    total_length = len(self.input_precipitation)

                    if self.mode == 0:
                        self.input_IR = self.input_IR[0: int(total_length * 0.8)]
                        self.output_IR = self.output_IR[0: int(total_length * 0.8)]


                    elif self.mode == 1:
                        self.input_IR = self.input_IR[int(total_length * 0.8): int(total_length * 0.9)]
                        self.output_IR = self.output_IR[int(total_length * 0.8): int(total_length * 0.9)]

                    elif self.mode == 2:
                        self.input_IR = self.input_IR[int(total_length * 0.9):]
                        self.output_IR = self.output_IR[int(total_length * 0.9):]
                    else:
                        raise Exception('Invalide mode, must be in [0,1,2]')
                print("IR Dataset input shape: ", self.input_IR.shape)
                print("IR Dataset output shape: ", self.output_IR.shape)

        if self.mode is not None:
            total_length = len(self.input_precipitation)

            if self.mode == 0:
                self.input_precipitation = self.input_precipitation[0: int(total_length * 0.8)]
                if not production_mode:
                    self.output_precipitation = self.output_precipitation[0: int(total_length * 0.8)]

            elif self.mode == 1:
                self.input_precipitation = self.input_precipitation[int(total_length * 0.8): int(total_length * 0.9)]
                if not production_mode:
                    self.output_precipitation = self.output_precipitation[
                                                int(total_length * 0.8): int(total_length * 0.9)]

            elif self.mode == 2:
                self.input_precipitation = self.input_precipitation[int(total_length * 0.9):]
                if not production_mode:
                    self.output_precipitation = self.output_precipitation[int(total_length * 0.9):]
            else:
                raise Exception('Invalide mode, must be in [0,1,2]')

        print("Precipitation Dataset input shape: ", self.input_precipitation.shape)
        if not production_mode:
            print("Precipitation Dataset output shape: ", self.output_precipitation.shape)

    # code obtained from https://stackoverflow.com/questions/19349410/how-to-pad-with-zeros-a-tensor-along-some-axis-python
    def symmetric_pad_array(self, input_array: np.ndarray, target_shape: tuple, pad_value: int) -> np.ndarray:
        for dim_in, dim_target in zip(input_array.shape, target_shape):
            if dim_target < dim_in:
                raise Exception("`target_shape` should be greater or equal than `input_array` shape for each axis.")

        pad_width = []
        for dim_in, dim_target in zip(input_array.shape, target_shape):
            if (dim_in - dim_target) % 2 == 0:
                pad_width.append((int(abs((dim_in - dim_target) / 2)), int(abs((dim_in - dim_target) / 2))))
            else:
                pad_width.append((int(abs((dim_in - dim_target) / 2)), (int(abs((dim_in - dim_target) / 2)) + 1)))

        return np.pad(input_array, pad_width, 'constant', constant_values=pad_value)

    def __getitem__(self, idx):
        if self.production_mode:
            if self.user_IR:
                return self.input_precipitation[idx], self.input_precipitation[idx], self.input_IR[idx], self.input_IR[
                    idx]
            else:
                return self.input_precipitation[idx], self.input_precipitation[idx]
        else:
            if self.user_IR:
                return self.input_precipitation[idx], self.output_precipitation[idx], self.input_IR[idx], \
                self.output_IR[idx]
            else:
                return self.input_precipitation[idx], self.output_precipitation[idx]

    def __len__(self):
        return len(self.input_precipitation)

    def get_input_timestamps(self):
        return self.input_timestamps

    def get_output_timestamps(self):
        return self.output_timestamps

    def get_output_precipitation(self):
        return self.output_precipitation[:, :, 0, :, :]


class ImergWAIRDataset(Dataset):
    def __init__(self, imerg_filename, ir_filename, forecast_steps, history_steps, normalize_data=False,
                 image_shape=(64, 64),
                 production_mode=False, mode=None):
        super(ImergWAIRDataset, self).__init__()
        self.imerg_filename = imerg_filename
        self.ir_filename = ir_filename
        self.normalize_data = normalize_data
        self.img_height = image_shape[0]
        self.img_width = image_shape[1]
        self.production_mode = production_mode
        self.mode = mode
        if self.ir_filename is not None:
            self.user_IR = True
        else:
            self.user_IR = False

        imerg_correction = 2
        with h5py.File(self.imerg_filename, 'r') as hf:
            precipitation_time_series = hf['precipitations'][:].astype(np.float32)[:-imerg_correction]
            timestamps = hf['timestamps'][:][:-imerg_correction]

            # crop the image to the desired shape(center crop)
            if self.img_height != precipitation_time_series.shape[1]:
                h_start = (precipitation_time_series.shape[1] - self.img_height) // 2
                precipitation_time_series = precipitation_time_series[:, h_start:h_start + self.img_height, :]

            if self.img_width != precipitation_time_series.shape[2]:
                w_start = (precipitation_time_series.shape[2] - self.img_width) // 2
                precipitation_time_series = precipitation_time_series[:, :, w_start:w_start + self.img_width]
            mean_imerg = hf['mean'][()]
            std_imerg = hf['std'][()]
            print("original shape", precipitation_time_series.shape)
            self.input_precipitation = sliding_window_view(precipitation_time_series,
                                                           window_shape=history_steps,
                                                           axis=0)[:-forecast_steps]

            self.input_timestamps = sliding_window_view(timestamps,
                                                        window_shape=history_steps,
                                                        axis=0)[:-forecast_steps]

            if not production_mode:
                self.output_precipitation = sliding_window_view(precipitation_time_series[history_steps:],
                                                                window_shape=forecast_steps,
                                                                axis=0)
                self.output_timestamps = sliding_window_view(timestamps[history_steps:],
                                                             window_shape=forecast_steps,
                                                             axis=0)

            # reshape to DGMR expected input
            self.input_precipitation = np.transpose(self.input_precipitation, (0, 3, 1, 2))
            if self.normalize_data == True:
                self.input_precipitation = (self.input_precipitation[:, :, None, :, :] - mean_imerg) / std_imerg
                self.input_precipitation = np.nan_to_num(self.input_precipitation, 0.)
            else:
                self.input_precipitation = self.input_precipitation[:, :, None, :, :]

            if not production_mode:
                self.output_precipitation = np.transpose(self.output_precipitation, (0, 3, 1, 2))
                if self.normalize_data == True:
                    self.output_precipitation = (self.output_precipitation[:, :, None, :, :] - mean_imerg) / std_imerg
                    self.output_precipitation = np.nan_to_num(self.output_precipitation, 0.)
                else:
                    self.output_precipitation = self.output_precipitation[:, :, None, :, :]

        if self.user_IR:
            ir_correction = 1
            with h5py.File(self.ir_filename, 'r') as hf:
                IR_time_series = hf['precipitations'][:].astype(np.float32)[1:-ir_correction]
                datetime_series = hf['timestamps'][:][1:-ir_correction]

                # mean_ir = hf['mean'][()]
                # std_ir = hf['std'][()]

                if self.img_height != IR_time_series.shape[1]:
                    h_start = (IR_time_series.shape[1] - self.img_height) // 2
                    IR_time_series = IR_time_series[:, h_start:h_start + self.img_height, :]

                if self.img_width != IR_time_series.shape[2]:
                    w_start = (IR_time_series.shape[2] - self.img_width) // 2
                    IR_time_series = IR_time_series[:, :, w_start:w_start + self.img_width]

                # print("IR original shape", IR_time_series.shape)
                history_steps_IR = history_steps * 2
                forecast_steps_IR = forecast_steps * 2

                input_IR_sample = sliding_window_view(IR_time_series,
                                                      window_shape=history_steps_IR,
                                                      axis=0)

                input_IR_timestamps_sample = sliding_window_view(datetime_series,
                                                                 window_shape=history_steps_IR,
                                                                 axis=0)

                input_IR_sample = input_IR_sample[:-forecast_steps_IR][::2]
                input_IR_timestamps_sample = input_IR_timestamps_sample[:-forecast_steps_IR][::2]

                output_IR_sample = sliding_window_view(IR_time_series[history_steps_IR:],
                                                       window_shape=forecast_steps_IR,
                                                       axis=0)[::2]
                output_IR_timestamps_sample = sliding_window_view(datetime_series[history_steps_IR:],
                                                                  window_shape=forecast_steps_IR,
                                                                  axis=0)[::2]

                # move 9 images from output IR to input IR (since we have up to 1 15h of IR in the past)
                # print(input_IR_sample.shape)
                # print(output_IR_sample.shape)

                self.input_IR = np.concatenate((input_IR_sample, output_IR_sample[:, :, :, 0:9]), axis=3)
                self.input_IR_timestamps_sample = np.concatenate(
                    (input_IR_timestamps_sample, output_IR_timestamps_sample[:, 0:9]), axis=1)

                self.output_IR = output_IR_sample[:, :, :, 9:]
                self.output_IR_timestamps_sample = output_IR_timestamps_sample[:, 9:]

                # reshape to DGMR expected input
                self.input_IR = np.transpose(self.input_IR, (0, 3, 1, 2))
                if self.normalize_data == True:
                    self.input_IR = (self.input_IR[:, -16:, None, :, :] - mean_ir) / std_ir
                    self.input_IR = np.nan_to_num(self.input_IR, 0.)
                else:
                    self.input_IR = self.input_IR[:, -16:, None, :, :]

                self.output_IR = np.transpose(self.output_IR, (0, 3, 1, 2))

                # normalize data and add an extra channel to incorporate neural network expected input
                if self.normalize_data == True:
                    self.output_IR = (self.output_IR[:, :, None, :, :] - mean_ir) / std_ir
                    self.output_IR = np.nan_to_num(self.output_IR, 0.)
                else:
                    self.output_IR = self.output_IR[:, :, None, :, :]

                if self.mode is not None:
                    total_length = len(self.input_precipitation)

                    if self.mode == 0:
                        self.input_IR = self.input_IR[0: int(total_length * 0.8)]
                        self.output_IR = self.output_IR[0: int(total_length * 0.8)]


                    elif self.mode == 1:
                        self.input_IR = self.input_IR[int(total_length * 0.8): int(total_length * 0.9)]
                        self.output_IR = self.output_IR[int(total_length * 0.8): int(total_length * 0.9)]

                    elif self.mode == 2:
                        self.input_IR = self.input_IR[int(total_length * 0.9):]
                        self.output_IR = self.output_IR[int(total_length * 0.9):]
                    else:
                        raise Exception('Invalide mode, must be in [0,1,2]')
                print("IR Dataset input shape: ", self.input_IR.shape)
                print("IR Dataset output shape: ", self.output_IR.shape)

        if self.mode is not None:
            total_length = len(self.input_precipitation)

            if self.mode == 0:
                self.input_precipitation = self.input_precipitation[0: int(total_length * 0.8)]
                if not production_mode:
                    self.output_precipitation = self.output_precipitation[0: int(total_length * 0.8)]

            elif self.mode == 1:
                self.input_precipitation = self.input_precipitation[int(total_length * 0.8): int(total_length * 0.9)]
                if not production_mode:
                    self.output_precipitation = self.output_precipitation[
                                                int(total_length * 0.8): int(total_length * 0.9)]

            elif self.mode == 2:
                self.input_precipitation = self.input_precipitation[int(total_length * 0.9):]
                if not production_mode:
                    self.output_precipitation = self.output_precipitation[int(total_length * 0.9):]
            else:
                raise Exception('Invalide mode, must be in [0,1,2]')

        print("Precipitation Dataset input shape: ", self.input_precipitation.shape)
        if not production_mode:
            print("Precipitation Dataset output shape: ", self.output_precipitation.shape)

    # code obtained from https://stackoverflow.com/questions/19349410/how-to-pad-with-zeros-a-tensor-along-some-axis-python
    def symmetric_pad_array(self, input_array: np.ndarray, target_shape: tuple, pad_value: int) -> np.ndarray:
        for dim_in, dim_target in zip(input_array.shape, target_shape):
            if dim_target < dim_in:
                raise Exception("`target_shape` should be greater or equal than `input_array` shape for each axis.")

        pad_width = []
        for dim_in, dim_target in zip(input_array.shape, target_shape):
            if (dim_in - dim_target) % 2 == 0:
                pad_width.append((int(abs((dim_in - dim_target) / 2)), int(abs((dim_in - dim_target) / 2))))
            else:
                pad_width.append((int(abs((dim_in - dim_target) / 2)), (int(abs((dim_in - dim_target) / 2)) + 1)))

        return np.pad(input_array, pad_width, 'constant', constant_values=pad_value)

    def __getitem__(self, idx):
        if self.production_mode:
            if self.user_IR:
                return self.input_precipitation[idx], _, self.input_IR[idx], _
            else:
                return self.input_precipitation[idx], _
        else:
            if self.user_IR:
                return self.input_precipitation[idx], self.output_precipitation[idx], self.input_IR[idx], \
                self.output_IR[idx]
            else:
                return self.input_precipitation[idx], self.output_precipitation[idx]

    def __len__(self):

        return len(self.input_precipitation)

    def get_input_timestamps(self):
        return self.input_timestamps

    def get_output_timestamps(self):
        return self.output_timestamps

    def get_output_precipitation(self):
        return self.output_precipitation[:, :, 0, :, :]


class IMERGDataModule(L.LightningDataModule):
    """
    Example of LightningDataModule for h5py IMERG dataset.
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
            forecast_steps=12,
            history_steps=12,
            imerg_filename="/home1/aaravamudan2014/data/ghana_imerg_2011_2020_Oct.h5",
            ir_filename="/home1/aaravamudan2014/data/wa_ir.h5",
            batch_size=32,
            image_shape=(64, 64),
            normalize_data=False,
            dataset=None,
            production_mode=False
    ):
        super().__init__()

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.imerg_filename = imerg_filename
        self.ir_filename = ir_filename
        self.dataloader_config = dict(
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            prefetch_factor=8,
            persistent_workers=True,
            # Disable automatic batching because dataset
            # returns complete batches.
            batch_size=None,
        )

        assert dataset is not None, "Enter dataset name"

        if dataset == 'wa':
            self.train_dataset = ImergWADataset(imerg_filename,
                                                ir_filename,
                                                forecast_steps,
                                                history_steps,
                                                normalize_data=normalize_data,
                                                image_shape=image_shape,
                                                production_mode=production_mode,
                                                mode = 0)
            self.val_dataset = ImergWADataset(imerg_filename,
                                                ir_filename,
                                                forecast_steps,
                                                history_steps,
                                                normalize_data=normalize_data,
                                                image_shape=image_shape,
                                                production_mode=production_mode,
                                                mode = 1)
            self.test_dataset = ImergWADataset(imerg_filename,
                                                ir_filename,
                                                forecast_steps,
                                                history_steps,
                                                normalize_data=normalize_data,
                                                image_shape=image_shape,
                                                production_mode=production_mode,
                                                mode = 2)
        elif dataset == 'wa_ir':
            """
            The IMERG and IR data is used as input.
            """
            self.train_dataset = ImergWAIRDataset(imerg_filename,
                                                  ir_filename,
                                                  forecast_steps,
                                                  history_steps,
                                                  normalize_data=normalize_data,
                                                  image_shape=image_shape,
                                                  production_mode=production_mode,
                                                  mode=0
                                                  )
            self.val_dataset = ImergWAIRDataset(imerg_filename,
                                                ir_filename,
                                                forecast_steps,
                                                history_steps,
                                                normalize_data=normalize_data,
                                                image_shape=image_shape,
                                                production_mode=production_mode,
                                                mode=1
                                                )
            self.test_dataset = ImergWAIRDataset(imerg_filename,
                                                 ir_filename,
                                                 forecast_steps,
                                                 history_steps,
                                                 normalize_data=normalize_data,
                                                 image_shape=image_shape,
                                                 production_mode=production_mode,
                                                 mode=2
                                                 )
        elif dataset == 'wa_ir_latent':
            self.train_dataset = IMERGDataModuleLatentDim(imerg_filename,
                                                  ir_filename,
                                                  forecast_steps,
                                                  history_steps,
                                                  normalize_data=normalize_data,
                                                  image_shape=image_shape,
                                                  production_mode=production_mode,
                                                  mode=0
                                                  )
            self.val_dataset = IMERGDataModuleLatentDim(imerg_filename,
                                                          ir_filename,
                                                          forecast_steps,
                                                          history_steps,
                                                          normalize_data=normalize_data,
                                                          image_shape=image_shape,
                                                          production_mode=production_mode,
                                                          mode=1
                                                          )
            self.test_dataset = IMERGDataModuleLatentDim(imerg_filename,
                                                          ir_filename,
                                                          forecast_steps,
                                                          history_steps,
                                                          normalize_data=normalize_data,
                                                          image_shape=image_shape,
                                                          production_mode=production_mode,
                                                          mode=2
                                                          )

    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=2)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=2)
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2)
        return dataloader

    def event_dataloader(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=2)
        return dataloader



"""unused """

# class ImergGhanaDataset(Dataset):
#     def __init__(self, precipitation_time_series, mean_imerg, std_imerg, ir_filename, forecast_steps, history_steps,
#                  normalize_data=False, image_shape=(64, 64)):
#         super(ImergGhanaDataset, self).__init__()
#         self.precipitation_time_series = precipitation_time_series
#         self.mean_imerg = mean_imerg
#         self.std_imerg = std_imerg
#         self.ir_filename = ir_filename
#         self.normalize_data = normalize_data
#         self.img_height = image_shape[0]
#         self.img_width = image_shape[1]

#         # crop the image to the desired shape(center crop)
#         if self.img_height != self.precipitation_time_series.shape[1]:
#             h_start = (self.precipitation_time_series.shape[1] - self.img_height) // 2
#             self.precipitation_time_series = self.precipitation_time_series[:, h_start:h_start + self.img_height, :]

#         if self.img_width != precipitation_time_series.shape[2]:
#             w_start = (self.precipitation_time_series.shape[2] - self.img_width) // 2
#             self.precipitation_time_series = self.precipitation_time_series[:, :, w_start:w_start + self.img_width]

#         print("original shape", precipitation_time_series.shape)
#         monthly_input_precipitation = sliding_window_view(precipitation_time_series,
#                                                           window_shape=history_steps,
#                                                           axis=0)

#         self.output_precipitation = sliding_window_view(precipitation_time_series[history_steps:],
#                                                         window_shape=forecast_steps,
#                                                         axis=0)
#         self.input_precipitation = monthly_input_precipitation[:-forecast_steps]

#         # reshape to DGMR expected input
#         self.input_precipitation = np.transpose(self.input_precipitation, (0, 3, 1, 2))
#         if self.normalize_data == True:
#             self.input_precipitation = (self.input_precipitation[:, :, None, :, :] - self.mean_imerg) / self.std_imerg
#             self.input_precipitation = np.nan_to_num(self.input_precipitation, 0.)
#         else:
#             self.input_precipitation = self.input_precipitation[:, :, None, :, :]

#         self.output_precipitation = np.transpose(self.output_precipitation, (0, 3, 1, 2))
#         if self.normalize_data == True:
#             self.output_precipitation = (self.output_precipitation[:, :, None, :, :] - self.mean_imerg) / self.std_imerg
#             self.output_precipitation = np.nan_to_num(self.output_precipitation, 0.)
#         else:
#             self.output_precipitation = self.output_precipitation[:, :, None, :, :]

#         print("Precipitation Dataset input shape: ", self.input_precipitation.shape)
#         print("Precipitation Dataset output shape: ", self.output_precipitation.shape)

#         # with h5py.File(self.ir_filename, 'r') as hf:
#         #     IR_time_series = hf['IRs'][:].astype(np.float32)
#         #     mean_ir = hf['mean'][()]
#         #     std_ir = hf['std'][()]
#         #     print("IR original shape", IR_time_series.shape)
#         #     IR_time_series = IR_time_series[start_index*31*48*2:end_index*31*48*2]
#         #     num_days_in_oct = 31
#         #     num_years = end_index - start_index
#         #     history_steps_IR =history_steps*2
#         #     forecast_steps_IR = forecast_steps*2
#         #     for i in range(num_years):
#         #         monthly_IR_time_series = IR_time_series[i*num_days_in_oct*48*2: (i+1)*num_days_in_oct*48*2]
#         #         monthly_input_IR= sliding_window_view(monthly_IR_time_series,
#         #                                                 window_shape=history_steps_IR,
#         #                                                 axis=0)
#         #         if i == 0:
#         #             output_IR_sample = sliding_window_view(monthly_IR_time_series[history_steps_IR:],
#         #                                                     window_shape=forecast_steps_IR,
#         #                                                     axis=0)[::2]
#         #             input_IR_sample = monthly_input_IR[:-forecast_steps_IR][::2]

#         #             # move 9 images from output IR to input IR (since we have up to 1 15h of IR in the past)
#         #             self.input_IR = np.concatenate((input_IR_sample, output_IR_sample[:,:,:,0:9]), axis=3)
#         #             self.output_IR = output_IR_sample[:,:,:,9:]
#         #         else:

#         #             output_IR_sample = sliding_window_view(monthly_IR_time_series[history_steps_IR:],
#         #                                                     window_shape=forecast_steps_IR,
#         #                                                     axis=0)[::2]
#         #             input_IR_sample = monthly_input_IR[:-forecast_steps_IR][::2]
#         #             # move 9 images from output IR to input IR (since we have up to 1 15h of IR in the past)
#         #             self.input_IR = np.concatenate((self.input_IR, np.concatenate((input_IR_sample, output_IR_sample[:,:,:,0:9]), axis=3)))
#         #             self.output_IR = np.concatenate((self.output_IR,output_IR_sample[:,:,:,9:]))

#         #     # reshape to DGMR expected input
#         #     self.input_IR = np.transpose(self.input_IR, (0, 3, 1, 2))
#         #     if self.normalize_data == True:
#         #         self.input_IR = (self.input_IR[:,-16:,None,:,:]-mean_ir)/std_ir
#         #         self.input_IR = np.nan_to_num(self.input_IR, 0.)
#         #     else:
#         #         self.input_IR = self.input_IR[:,-16:,None,:,:]

#         #     self.output_IR = np.transpose(self.output_IR, (0, 3, 1, 2))
#         #     if self.normalize_data == True:
#         #         self.output_IR = (self.output_IR[:,:,None,:,:]-mean_ir)/std_ir
#         #         self.output_IR = np.nan_to_num(self.output_IR, 0.)
#         #     else:
#         #         self.output_IR = self.output_IR[:,:,None,:,:]

#         #     print("IR Dataset input shape: ", self.input_IR.shape)
#         #     print("IR Dataset output shape: ", self.output_IR.shape)

#     # code obtained from https://stackoverflow.com/questions/19349410/how-to-pad-with-zeros-a-tensor-along-some-axis-python
#     def symmetric_pad_array(self, input_array: np.ndarray, target_shape: tuple, pad_value: int) -> np.ndarray:
#         for dim_in, dim_target in zip(input_array.shape, target_shape):
#             if dim_target < dim_in:
#                 raise Exception("`target_shape` should be greater or equal than `input_array` shape for each axis.")

#         pad_width = []
#         for dim_in, dim_target in zip(input_array.shape, target_shape):
#             if (dim_in - dim_target) % 2 == 0:
#                 pad_width.append((int(abs((dim_in - dim_target) / 2)), int(abs((dim_in - dim_target) / 2))))
#             else:
#                 pad_width.append((int(abs((dim_in - dim_target) / 2)), (int(abs((dim_in - dim_target) / 2)) + 1)))

#         return np.pad(input_array, pad_width, 'constant', constant_values=pad_value)

#     def __getitem__(self, idx):
#         return self.input_precipitation[idx], self.output_precipitation[idx]

#     def __len__(self):
#         return len(self.output_precipitation)


# class ImergGhanaMonthlyDataset(Dataset):
#     def __init__(self, precipitation_time_series, IR_time_series, start_index, end_index, forecast_steps, history_steps,
#                  image_shape=(64, 64)):
#         super(ImergGhanaMonthlyDataset, self).__init__()
#         self.precipitation_time_series = precipitation_time_series
#         # self.mean_imerg = mean_imerg
#         # self.std_imerg = std_imerg
#         self.IR_time_series = IR_time_series
#         # self.normalize_data = normalize_data
#         self.img_height = image_shape[0]
#         self.img_width = image_shape[1]

#         # crop the image to the desired shape(center crop)

#         if self.img_height != self.precipitation_time_series.shape[1]:
#             h_start = (self.precipitation_time_series.shape[1] - self.img_height) // 2
#             self.precipitation_time_series = self.precipitation_time_series[:, h_start:h_start + self.img_height, :]

#         if self.img_width != precipitation_time_series.shape[2]:
#             w_start = (precipitation_time_series.shape[2] - self.img_width) // 2
#             precipitation_time_series = precipitation_time_series[:, :, w_start:w_start + self.img_width]

#         print("original shape", precipitation_time_series.shape)
#         precipitation_time_series = precipitation_time_series[start_index * 31 * 48:end_index * 31 * 48]
#         num_days_in_oct = 31
#         num_years = end_index - start_index

#         for i in range(num_years):
#             monthly_precipitation_time_series = precipitation_time_series[
#                                                 i * num_days_in_oct * 48: (i + 1) * num_days_in_oct * 48]

#             monthly_input_precipitation = sliding_window_view(monthly_precipitation_time_series,
#                                                               window_shape=history_steps,
#                                                               axis=0)
#             if i == 0:
#                 self.output_precipitation = sliding_window_view(monthly_precipitation_time_series[history_steps:],
#                                                                 window_shape=forecast_steps,
#                                                                 axis=0)
#                 self.input_precipitation = monthly_input_precipitation[:-forecast_steps]
#             else:
#                 self.output_precipitation = np.concatenate(
#                     (self.output_precipitation, sliding_window_view(monthly_precipitation_time_series[history_steps:],
#                                                                     window_shape=forecast_steps,
#                                                                     axis=0)))
#                 self.input_precipitation = np.concatenate(
#                     (self.input_precipitation, monthly_input_precipitation[:-forecast_steps]))

#         # reshape to DGMR expected input
#         self.input_precipitation = np.transpose(self.input_precipitation, (0, 3, 1, 2))
#         # if self.normalize_data == True:
#         #     self.input_precipitation = (self.input_precipitation[:,:,None,:,:]-self.mean_imerg)/self.std_imerg
#         #     self.input_precipitation = np.nan_to_num(self.input_precipitation, 0.)
#         # else:
#         self.input_precipitation = self.input_precipitation[:, :, None, :, :]

#         self.output_precipitation = np.transpose(self.output_precipitation, (0, 3, 1, 2))
#         # if self.normalize_data == True:
#         #     self.output_precipitation = (self.output_precipitation[:,:,None,:,:]-self.mean_imerg)/self.std_imerg
#         #     self.output_precipitation = np.nan_to_num(self.output_precipitation, 0.)
#         # else:
#         self.output_precipitation = self.output_precipitation[:, :, None, :, :]

#         print("Precipitation Dataset input shape: ", self.input_precipitation.shape)
#         print("Precipitation Dataset output shape: ", self.output_precipitation.shape)

#         print("IR original shape", IR_time_series.shape)
#         IR_time_series = IR_time_series[start_index * 31 * 48 * 2:end_index * 31 * 48 * 2]
#         num_days_in_oct = 31
#         num_years = end_index - start_index
#         history_steps_IR = history_steps * 2
#         forecast_steps_IR = forecast_steps * 2
#         for i in range(num_years):
#             monthly_IR_time_series = IR_time_series[i * num_days_in_oct * 48 * 2: (i + 1) * num_days_in_oct * 48 * 2]
#             monthly_input_IR = sliding_window_view(monthly_IR_time_series,
#                                                    window_shape=history_steps_IR,
#                                                    axis=0)
#             if i == 0:
#                 output_IR_sample = sliding_window_view(monthly_IR_time_series[history_steps_IR:],
#                                                        window_shape=forecast_steps_IR,
#                                                        axis=0)[::2]
#                 input_IR_sample = monthly_input_IR[:-forecast_steps_IR][::2]

#                 # move 9 images from output IR to input IR (since we have up to 1 15h of IR in the past)
#                 self.input_IR = np.concatenate((input_IR_sample, output_IR_sample[:, :, :, 0:9]), axis=3)
#                 self.output_IR = output_IR_sample[:, :, :, 9:]
#             else:

#                 output_IR_sample = sliding_window_view(monthly_IR_time_series[history_steps_IR:],
#                                                        window_shape=forecast_steps_IR,
#                                                        axis=0)[::2]
#                 input_IR_sample = monthly_input_IR[:-forecast_steps_IR][::2]
#                 # move 9 images from output IR to input IR (since we have up to 1 15h of IR in the past)
#                 self.input_IR = np.concatenate(
#                     (self.input_IR, np.concatenate((input_IR_sample, output_IR_sample[:, :, :, 0:9]), axis=3)))
#                 self.output_IR = np.concatenate((self.output_IR, output_IR_sample[:, :, :, 9:]))

#         # reshape to DGMR expected input
#         self.input_IR = np.transpose(self.input_IR, (0, 3, 1, 2))
#         # if self.normalize_data == True:
#         #     self.input_IR = (self.input_IR[:,-16:,None,:,:]-mean_ir)/std_ir
#         #     self.input_IR = np.nan_to_num(self.input_IR, 0.)
#         # else:
#         self.input_IR = (self.input_IR[:, -16:, None, :, :] / 343.1587) * 53.2
#         self.input_IR = np.nan_to_num(self.input_IR, 0.)

#         self.output_IR = np.transpose(self.output_IR, (0, 3, 1, 2))
#         # if self.normalize_data == True:
#         #     self.output_IR = (self.output_IR[:,:,None,:,:]-mean_ir)/std_ir
#         #     self.output_IR = np.nan_to_num(self.output_IR, 0.)
#         # else:
#         self.output_IR = self.output_IR[:, :, None, :, :]

#         print("IR Dataset input shape: ", self.input_IR.shape)
#         print("IR Dataset output shape: ", self.output_IR.shape)

#     # code obtained from https://stackoverflow.com/questions/19349410/how-to-pad-with-zeros-a-tensor-along-some-axis-python

#     def symmetric_pad_array(self, input_array: np.ndarray, target_shape: tuple, pad_value: int) -> np.ndarray:
        
#         for dim_in, dim_target in zip(input_array.shape, target_shape):
#             if dim_target < dim_in:
#                 raise Exception("`target_shape` should be greater or equal than `input_array` shape for each axis.")

#         pad_width = []
#         for dim_in, dim_target in zip(input_array.shape, target_shape):
#             if (dim_in - dim_target) % 2 == 0:
#                 pad_width.append((int(abs((dim_in - dim_target) / 2)), int(abs((dim_in - dim_target) / 2))))
#             else:
#                 pad_width.append((int(abs((dim_in - dim_target) / 2)), (int(abs((dim_in - dim_target) / 2)) + 1)))

#         return np.pad(input_array, pad_width, 'constant', constant_values=pad_value)

#     def __getitem__(self, idx):
#         return self.input_precipitation[idx], self.input_IR[idx], self.output_precipitation[idx]

#     def __len__(self):
#         return len(self.output_precipitation)