from servir.methods.ExtrapolationMethods.naive_persistence import naive_persistence
from servir.methods.ExtrapolationMethods.extrapolation_methods import langragian_persistance, steps, linda
from servir.methods.ConvLSTM.ConvLSTM import ConvLSTM
from servir.core.distribution import get_dist_info
import h5py 
import datetime
import numpy as np
from servir.utils.config_utils import load_config
import functools
import torch
from servir.utils.data_provider import ImergDataset, IMERGDataModule
from servir.methods.dgmr.dgmr import DGMR

class ModelPicker:
    def __init__(self, model_type, model_config_location, model_save_location=None, use_gpu=False) -> None:
        self.model_type = model_type
        self.model_config_location = model_config_location
        self.model_save_location = model_save_location
        self.input_precip = None
        self.use_gpu = use_gpu
            
    
        
    def load_model(self):
        self.config = load_config(self.model_config_location)
        if self.model_type == 'naive':
            self.prediction_function = lambda y: naive_persistence(y, output_sequence_length=self.config['out_seq_length'])
        elif self.model_type == 'lagrangian':
            self.prediction_function = lambda y: langragian_persistance(y, timesteps=self.config['out_seq_length'])
        elif self.model_type == 'steps':
            self.prediction_function = lambda y: steps(y, timesteps=self.config['out_seq_length'], n_cascade_levels=self.config['n_cascade_levels'], n_ens_members=self.config['n_ens_members'])
        elif self.model_type == 'linda':
            self.prediction_function = lambda y: linda(y, timesteps=self.config['out_seq_length'], max_num_features=self.config['max_num_features'], add_perturbations=self.config['add_perturbations'])
        elif self.model_type == 'convlstm':
            if self.use_gpu and torch.cuda.is_available(): 
                device = torch.device('cuda:0')
            else:
                device = torch.device('cpu')
            self.config['device'] = device
            self.config['rank'], self.config['world_size'] = get_dist_info()
            self.config['relu_last'] = True
            conv_lstm_nodel =ConvLSTM(self.config) 
            if not self.use_gpu:
                conv_lstm_nodel.model.load_state_dict(torch.load(self.model_save_location, map_location=torch.device('cpu')))
            else:
                conv_lstm_nodel.model.load_state_dict(torch.load(self.model_save_location))

            self.max_rainfall_intensity = 60
            self.input_precip =  self.input_precip / self.max_rainfall_intensity
            # add batch and channel dimension to input. From [T, H, W] to [B, T, C, H, W]
            self.input_precip = np.expand_dims(self.input_precip, axis=(0,2))
            
            Y = torch.tensor(np.zeros(self.input_precip.shape), dtype=torch.float32, device=device)
            Y = torch.tensor(Y, dtype=torch.float32, device=device)
            
            self.prediction_function = lambda x: conv_lstm_nodel._predict(torch.tensor(x, dtype=torch.float32, device=device),Y)
        elif self.model_type == 'dgmr':
            if self.use_gpu and torch.cuda.is_available(): 
                device = torch.device('cuda:0')
            else:
                device = torch.device('cpu')
            
            # datamodule = IMERGDataModule(forecast_steps = self.config['out_seq_length'],
            # history_steps = self.config['in_seq_length'],
            # imerg_filename = self.input_h5_filename,
            # ir_filename = None,
            # batch_size = self.config['batch_size'],
            # )
        
            # input_channels = self.config['input_channels']
            # output_shape = self.config['img_shape']
            # model = DGMR(
            #     forecast_steps=self.config['out_seq_length'],
            #     input_channels=input_channels,
            #     output_shape=output_shape,
            #     latent_channels=self.config['latent_channels'], 
            #     context_channels=self.config['context_channels'], 
            #     num_samples=5,
            #     visualize=False
            # )
            model = DGMR.load_from_checkpoint(self.model_save_location, map_location=device)
            
            self.input_precip = self.input_precip.astype(np.float32)
            img_height = self.config['img_shape'][0]
            img_width = self.config['img_shape'][1]
            
            if img_height != self.input_precip.shape[1]:
                h_start = (self.input_precip.shape[1] - img_height) // 2
                self.input_precip = self.input_precip[:, h_start:h_start+img_height, :]
            
            if img_width != self.input_precip.shape[2]:
                w_start = (self.input_precip.shape[2] - img_width) // 2
                self.input_precip = self.input_precip[:, :, w_start:w_start+img_width]


            self.input_precip = self.input_precip[-self.config['in_seq_length']:, :, :]
            # self.input_precip = np.transpose(self.input_precip, (2, 0, 1))
            self.input_precip = torch.tensor(self.input_precip[:,None,:,:])
            self.input_precip = torch.tensor(self.input_precip[None,:,:,:,:])
            
            self.prediction_function = model
            
            
    
    def load_data(self, input_h5_filename):
        self.input_h5_filename = input_h5_filename
        # Load the input precipitations
        with h5py.File(input_h5_filename, 'r') as hf:
            input_precip = hf['precipitations'][:]
            input_dt = hf['timestamps'][:]
            input_dt = np.array([datetime.datetime.strptime(x.decode('utf-8'), '%Y-%m-%d %H:%M:%S') for x in input_dt])
        
        self.input_precip = input_precip
        self.input_dt = input_dt


    def predict(self):
        assert self.input_precip is not None, "Data needs to be loaded first, please make a call to load_data()"
        if self.model_type in ['naive', 'linda', 'steps', 'lagrangian']:
            return self.prediction_function(self.input_precip)
        elif self.model_type in ['convlstm']:
            # convert to tensor 
            with torch.no_grad():
                pred_Y = self.prediction_function(self.input_precip)
            # convert to numpy
            pred_Y = pred_Y.cpu().numpy()
            # reduce batch and channel dimension
            pred_Y = np.squeeze(pred_Y, axis=(0,2))
            # convert to original samples
            pred_Y = pred_Y * self.max_rainfall_intensity
            return pred_Y
        elif self.model_type in ['dgmr']:
            pred_out_images = self.prediction_function(self.input_precip)
            return pred_out_images.detach().numpy()
    
    def save_output(self, output_h5_filename, output_precipitation):
        
        output_dt = [self.input_dt[-1] + datetime.timedelta(minutes=30*(k+1)) for k in range(self.config['out_seq_length'])]
        output_dt_str = [x.strftime('%Y-%m-%d %H:%M:%S') for x in output_dt]

        # save results to h5py file
        with h5py.File(output_h5_filename,'w') as hf:
            hf.create_dataset('precipitations', data=output_precipitation)
            hf.create_dataset('timestamps', data=output_dt_str)