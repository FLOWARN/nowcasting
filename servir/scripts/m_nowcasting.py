import os
import sys
import h5py
import datetime

import numpy as np  
import torch
from servir.core.model_picker import ModelPicker

param_dict_of_dicts = {'naive':{'model_type': 'naive',
                                    'config_path': 'ML/servir_nowcasting_examples/configs/wa_imerg/naive_persistence.py', 
                                    'model_save_path': None, 
                                    'input_h5_fname': 'ML/servir_nowcasting_examples/temp/input_imerg.h5', 
                                    'output_h5_fname':'ML/servir_nowcasting_examples/temp/output_naive_persistence.h5', 
                                    'use_gpu': False
                                    },
                           'lagrangian': {'model_type': 'lagrangian',
                                    'config_path': 'ML/servir_nowcasting_examples/configs/wa_imerg/lagrangian_persistence.py', 
                                    'model_save_path': None, 
                                    'input_h5_fname': 'ML/servir_nowcasting_examples/temp/input_imerg.h5', 
                                    'output_h5_fname':'ML/servir_nowcasting_examples/temp/output_lagrangian_persistence.h5', 
                                    'use_gpu': False
                           },
                           'linda': {'model_type': 'linda',
                                    'config_path': 'ML/servir_nowcasting_examples/configs/wa_imerg/LINDA.py', 
                                    'model_save_path': None, 
                                    'input_h5_fname': 'ML/servir_nowcasting_examples/temp/input_imerg.h5', 
                                    'output_h5_fname':'ML/servir_nowcasting_examples/temp/output_linda.h5', 
                                    'use_gpu': False
                           },
                           'steps': {'model_type': 'steps',
                                    'config_path': 'ML/servir_nowcasting_examples/configs/wa_imerg/PySTEPS.py', 
                                    'model_save_path': None, 
                                    'input_h5_fname': 'ML/servir_nowcasting_examples/temp/input_imerg.h5', 
                                    'output_h5_fname':'ML/servir_nowcasting_examples/temp/output_steps.h5', 
                                    'use_gpu': False
                           },
                           'convlstm': {'model_type': 'convlstm',
                                    'config_path': 'ML/servir_nowcasting_examples/configs/wa_imerg/ConvLSTM.py', 
                                    'model_save_path': 'ML/servir_nowcasting_examples/temp/imerg_only_mse_params.pth', 
                                    'input_h5_fname': 'ML/servir_nowcasting_examples/temp/input_imerg.h5', 
                                    'output_h5_fname':'ML/servir_nowcasting_examples/temp/output_convlstm.h5',
                                    'use_gpu': False
                           },
                           'dgmr': {'model_type': 'dgmr',
                                    'config_path': 'configs/wa_imerg/DGMR.py', 
                                    'model_save_path': 'temp/dgmr-None.ckpt', 
                                    'input_h5_fname': 'temp/input_imerg.h5', 
                                    'output_h5_fname':'temp/output_dgmr.h5',
                                    'use_gpu': False
                           }
                        }
    
    
def nowcast(param_dict):
    """
    Function to run nowcasting using the model picker class.
    The function loads the data from the input h5 file, loads the model, and then makes predictions.
    The predictions are then saved to the output h5 file.

    Args:
        param_dict (_type_): _description_
    """
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
    
    model_picker.load_model(get_ensemble=False)
    
    predictions = model_picker.predict()

    model_picker.save_output(output_h5_fname, predictions, num_predictions = 1)


def load_default_params_for_model(model_name):
    """
    Function to load the default parameters for a given model.
    The function loads the parameters from the param_dict_of_dicts dictionary.
    The function returns the parameters for the given model name.
    The function raises an assertion error if the model name is not in the dictionary.


    Args:
        model_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    assert model_name in list(param_dict_of_dicts.keys())
    return param_dict_of_dicts[model_name]
    

def parse_parameters():
    """
    Function to parse the parameters for the nowcasting function.
    The function uses argparse to parse the parameters.
    The function returns a dictionary of the parameters.

    Returns:
        _type_: _description_
    """
    import argparse
    
    
    parser=argparse.ArgumentParser()
    
    default_model_name = 'dgmr'
    print(param_dict_of_dicts[default_model_name])
    parser.add_argument("--model_type", help="Model Type (naive, lagrangian, linda, steps, convlstm, dgmr)", default=param_dict_of_dicts[default_model_name]['model_type'])
    parser.add_argument("--config_path", help="Path for config files for model", default=param_dict_of_dicts[default_model_name]['config_path'])
    parser.add_argument("--model_save_path", help="Model save path (relevant if the underlying model requires a saved object)", default=param_dict_of_dicts[default_model_name]['model_save_path'])
    parser.add_argument("--use_gpu", help="Whether or not to use GPU", default=param_dict_of_dicts[default_model_name]['use_gpu'], action="store_true")
    parser.add_argument("--input_h5_fname", help="file name to (optionally create) use for the input image time series", default=param_dict_of_dicts[default_model_name]['input_h5_fname'])
    parser.add_argument("--output_h5_fname", help="file name to save outputs to", default=param_dict_of_dicts[default_model_name]['output_h5_fname'])
    
    args=parser.parse_args()

    param_dict = vars(args)

    return param_dict
    
if __name__ == "__main__":
            
    param_dict = parse_parameters()
    nowcast(param_dict=param_dict)