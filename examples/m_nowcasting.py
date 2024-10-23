import os
import sys
import h5py
import datetime

import numpy as np  
import torch
from servir.core.model_picker import ModelPicker


def nowcast(param_dict):
    model_type = param_dict['model_type']
    model_config_path = param_dict['config_path']
    model_save_path = param_dict['model_save_path']
    input_h5_fname = param_dict['input_h5_fname']
    output_h5_fname = param_dict['output_h5_fname']
    use_gpu = param_dict['use_gpu']
    model_picker = ModelPicker(model_type=model_type, 
                               model_config_location=model_config_path, 
                               model_save_location=model_save_path,
                               use_gpu=use_gpu)
    model_picker.load_data(input_h5_fname)
    
    model_picker.load_model()
    
    predictions = model_picker.predict()
    
    model_picker.save_output(output_h5_fname, predictions)

    

def parse_parameters():
    import argparse
    parser=argparse.ArgumentParser()
    
    parser.add_argument("--model_type", help="Model Type (naive, lagrangian, linda, steps, convlstm, dgmr)", default='dgmr')
    # parser.add_argument("--servir_path", help="Model Type (naive, lagrangian, linda, steps, convlstm, dgmr)", default='/Users/akshayaravamudan/Desktop/SERVIR/nowcasting/')
    parser.add_argument("--config_path", help="Path for config files for model", default='../configs/gh_imerg/DGMR.py')
    parser.add_argument("--model_save_path", help="Model save path (relevant if the underlying model requires a saved object)", default='temp/DGMRBestUnnormalized.pth')
    parser.add_argument("--use_gpu", help="Whether or not to use GPU", default=False, action="store_true")
    parser.add_argument("--input_h5_fname", help="file name to (optionally create) use for the input image time series", default='temp/input_imerg.h5')
    parser.add_argument("--output_h5_fname", help="file name to save outputs to", default='temp/output_imerg_dgmr.h5')
    
    args=parser.parse_args()

    param_dict = vars(args)
    
    return param_dict

    # model_type = 'lagrangian'
    # servir_path ='/home/cc/projects/nowcasting'
    # config_path = '/home/cc/projects/nowcasting/temp/ConvLSTM_Config.py'
    # para_dict_fpath = '/home/cc/projects/nowcasting/temp/imerg_only_mse_params.pth'
    # use_gpu = False

    # input_h5_fname = '/home/cc/projects/nowcasting/temp/input_imerg.h5'
    # output_h5_fname = '/home/cc/projects/nowcasting/temp/output_imerg.h5'
    
if __name__ == "__main__":
    param_dict = parse_parameters()
    nowcast(param_dict=param_dict)