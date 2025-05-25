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
from servir.utils.data_provider import IMERGDataModule
from servir.methods.dgmr.dgmr import DGMR
from servir.methods.dgmr_ir.dgmr_ir import DGMR_IR
import os

class ModelPicker:
    """
    Class to load the model based on the model type and configuration file.
    The class also loads the data and makes predictions based on the loaded model.
    The class also saves the output to a h5py file.
    Args:
        model_type (str): type of the model to be loaded. Options are 'naive', 'steps', 'linda', 'convlstm', 'dgmr', 'dgmr_ir'.
        model_config_location (str): location of the model configuration file.
        model_save_location (str): location of the saved model file.
        use_gpu (bool): whether to use GPU for prediction. Defaults to False.
    """
    def __init__(self, model_type, model_config_location, model_save_location=None, use_gpu=False) -> None:
        self.model_type = model_type
        self.model_config_location = model_config_location
        self.model_save_location = model_save_location
        self.input_precip = None
        self.use_gpu = use_gpu
            

    def load_model(self, get_ensemble=True):
        """
        Function to load the model based on the model type and configuration file.

        Args:
            get_ensemble (bool, optional): _description_. Defaults to True.
        """
        self.config = load_config(self.model_config_location)
        
        if not get_ensemble:
            n_ens_members = 1
        else:
            n_ens_members = self.config['n_ens_members']
            
        if self.model_type == 'naive':
            self.prediction_function = lambda y: naive_persistence(y, output_sequence_length=self.config['out_seq_length'])
        elif self.model_type == 'lagrangian':
            self.prediction_function = lambda y: langragian_persistance(y, timesteps=self.config['out_seq_length'])
        elif self.model_type == 'steps':
            self.prediction_function = lambda y: steps(y, timesteps=self.config['out_seq_length'], n_cascade_levels=self.config['n_cascade_levels'], n_ens_members=n_ens_members, return_output=True)
        elif self.model_type == 'linda':
            self.prediction_function = lambda y: linda(y, timesteps=self.config['out_seq_length'], max_num_features=self.config['max_num_features'], n_ens_members=n_ens_members, add_perturbations=self.config['add_perturbations'], return_output=True)
        elif self.model_type == 'convlstm':
            if self.use_gpu and torch.cuda.is_available(): 
                device = torch.device('cuda:0')
            else:
                device = torch.device('cpu')
            self.config['device'] = device
            self.config['rank'], self.config['world_size'] = get_dist_info()
            self.config['relu_last'] = True
            conv_lstm_model =ConvLSTM(self.config) 
            if not self.use_gpu:
                conv_lstm_model.model.load_state_dict(torch.load(self.model_save_location, map_location=torch.device('cpu'), weights_only=False))
            else:
                conv_lstm_model.model.load_state_dict(torch.load(self.model_save_location))

            self.prediction_function = lambda x,y: conv_lstm_model._predict(torch.tensor(x, dtype=torch.float32, device=device),y)
        
        elif self.model_type == 'dgmr':
            if self.use_gpu and torch.cuda.is_available(): 
                device = torch.device('cuda:0')
            else:
                device = torch.device('cpu')
            
            model = DGMR.load_from_checkpoint(self.model_save_location, map_location=device)
            self.prediction_function = lambda y: model.predict_ensemble(y, n_ens_members= n_ens_members)
        elif self.model_type == 'dgmr_ir':
            if self.use_gpu and torch.cuda.is_available(): 
                device = torch.device('cuda:0')
            else:
                device = torch.device('cpu')
            
            model = DGMR_IR.load_from_checkpoint(self.model_save_location, map_location=device)    
            self.prediction_function = lambda y, y_ir: model.predict_ensemble(y, y_ir, n_ens_members = n_ens_members)
            
    def load_data(self, input_h5_filename):
        """
        Function to load the data from the h5py file. The function loads the precipitation data and the timestamps.
        The precipitation data is loaded as a numpy array and the timestamps are loaded as a list of datetime objects.
        The function also converts the timestamps to string format.


        Args:
            input_h5_filename (_type_): _description_
        """
        self.input_h5_filename = input_h5_filename
        # Load the input precipitations
        with h5py.File(input_h5_filename, 'r') as hf:
            input_precip = hf['precipitations'][:]
            input_dt = hf['timestamps'][:]
            input_dt = np.array([datetime.datetime.strptime(x.decode('utf-8'), '%Y-%m-%d %H:%M:%S') for x in input_dt])
        
        self.input_precip = input_precip
        self.input_dt = input_dt

    def predict(self, samples = None, samples_ir = None):
        """
        Function to make predictions based on the loaded model and data. The function takes the input precipitation data and
        the timestamps as input. The function also takes the input IR data as input. The function returns the predicted precipitation data.
        

        Args:
            samples (_type_, optional): _description_. Defaults to None.
            samples_ir (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if samples is not None:
            self.input_precip = samples
        
        self.input_ir = samples_ir
            
        assert self.input_precip is not None, "Data needs to be loaded first, please make a call to load_data()"
        if self.model_type in ['naive', 'linda', 'steps', 'lagrangian']:
            return self.prediction_function(self.input_precip[-self.config['in_seq_length']:])
        elif self.model_type in ['convlstm']:
            self.img_height = self.config['img_height']
            self.img_width = self.config['img_width']
            self.max_rainfall_intensity = 60
            input_precip =  self.input_precip / self.max_rainfall_intensity
            self.img_height = self.config['img_height']
            self.img_width = self.config['img_width']
            
            if self.img_height != input_precip.shape[1]:
                h_start = (input_precip.shape[1] - self.img_height) // 2
                input_precip = input_precip[:, h_start:h_start+self.img_height, :]
        
            if self.img_width != input_precip.shape[2]:
                w_start = (input_precip.shape[2] - self.img_width) // 2
                input_precip = input_precip[:, :, w_start:w_start+self.img_width]

            # add batch and channel dimension to input. From [T, H, W] to [B, T, C, H, W]
            input_precip = np.expand_dims(input_precip, axis=(0,2))
            
            Y = torch.tensor(np.zeros(input_precip.shape), dtype=torch.float32, device=self.config['device'])
            Y = torch.tensor(Y, dtype=torch.float32, device= self.config['device'])
            
            # convert to tensor
            with torch.no_grad():
                pred_Y = self.prediction_function(input_precip[:,-self.config['in_seq_length']:,:,:,:], Y)
            # convert to numpy

            pred_Y = pred_Y.cpu().numpy()
            # reduce batch and channel dimension
            pred_Y = np.squeeze(pred_Y, axis=(0,2))
            # convert to original samples
            pred_Y = pred_Y * self.max_rainfall_intensity
            return pred_Y
        elif self.model_type in ['dgmr']:
            if samples is None:
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
            else:
                self.input_precip = samples
                pred_out_images = self.prediction_function(self.input_precip)
                pred_out_images = [x.detach().numpy() for x in  pred_out_images]
                return pred_out_images
        elif self.model_type in ['dgmr_ir']:
            if samples is None:
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
                
            else:
                self.input_precip = samples
                pred_out_images = self.prediction_function(self.input_precip, self.input_ir)
                pred_out_images = [x.detach().numpy() for x in  pred_out_images]
                return pred_out_images
            
            pred_out_images = self.prediction_function(self.input_precip)
            pred_out_images = [x.detach().numpy() for x in  pred_out_images]
            
            return pred_out_images

    def save_output(self, output_h5_filename, output_precipitation, num_predictions, dataset = 'IMERG'):
        """
        Function to save the output to a h5py file. The function takes the output precipitation data and the timestamps as input.
        The function also takes the output h5py filename as input. The function returns the predicted precipitation data.
        The function also deletes any existing file with the same name.

        Args:
            output_h5_filename (_type_): _description_
            output_precipitation (_type_): _description_
            num_predictions (_type_): _description_
            dataset (str, optional): _description_. Defaults to 'IMERG'.

        Raises:
            Exception: _description_
        """
        if dataset == 'IMERG':
            output_dt = [self.input_dt[-1] + datetime.timedelta(minutes=30*(k+1)) for k in range(self.config['out_seq_length'])]
        elif dataset == 'PDIR':
            output_dt = [self.input_dt[-1] + datetime.timedelta(minutes=60*(k+1)) for k in range(self.config['out_seq_length'])]
        else:
            raise Exception('Wrong dataset name')
        output_dt_str = [x.strftime('%Y-%m-%d %H:%M:%S') for x in output_dt]
        if num_predictions == 1:
            output_precipitation = output_precipitation[None, :, :, :]
        # delete any existing file
        if os.path.isfile(output_h5_filename):
            with h5py.File(output_h5_filename,  "a") as f:
                for index, prediction in enumerate(output_precipitation):
                    if num_predictions == 1:
                        del f['precipitations']
                        del f['timestamps']
                    else:
                        del f[str(index) +'precipitations']
                        del f[str(index) +'timestamps']
                

        # save results to h5py file
        with h5py.File(output_h5_filename,'w') as hf:
            for index, prediction in enumerate(output_precipitation):
                # hf.create_dataset(str(index + 1), data = {'precipitations': prediction, 'timestamps': output_dt_str})
                if num_predictions == 1:
                    hf.create_dataset('precipitations', data=prediction)
                    hf.create_dataset('timestamps', data=output_dt_str)  
                else:                  
                    hf.create_dataset(str(index) + 'precipitations', data=prediction)
                    hf.create_dataset(str(index) + 'timestamps', data=output_dt_str)