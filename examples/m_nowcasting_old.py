import os
import sys
import h5py
import datetime

import numpy as np  
import torch

servir_path = sys.argv[1]

config_path = sys.argv[2]
para_dict_fpath = sys.argv[3]   
use_gpu = sys.argv[4]

input_h5_fname = sys.argv[5]
output_h5_fname = sys.argv[6]


# servir_path ='/home/cc/projects/nowcasting' 

# config_path = '/home/cc/projects/nowcasting/temp/ConvLSTM_Config.py'
# para_dict_fpath = '/home/cc/projects/nowcasting/temp/imerg_only_mse_params.pth'
# use_gpu = False

# input_h5_fname = '/home/cc/projects/nowcasting/temp/input_imerg.h5'
# output_h5_fname = '/home/cc/projects/nowcasting/temp/output_imerg.h5'

sys.path.append(servir_path)

from servir.utils.config_utils import load_config
from servir.methods.ConvLSTM import ConvLSTM
from servir.core.distribution import get_dist_info


# Load configuration file
config = load_config(config_path)
config['steps_per_epoch'] = 1
##==================Setup Method=====================##
# get device

if use_gpu and torch.cuda.is_available(): 
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

config['device'] = device
config['rank'], config['world_size'] = get_dist_info()
config['relu_last'] = True


# setup method
method = ConvLSTM(config)
# Loads best model’s parameter dictionary 
if device.type == 'cpu':
    method.model.load_state_dict(torch.load(para_dict_fpath, map_location=torch.device('cpu')))
else:
    method.model.load_state_dict(torch.load(para_dict_fpath))

# method.model.load_state_dict(torch.load(para_dict_fpath))

# Load the predictions
with h5py.File(input_h5_fname, 'r') as hf:
    input = hf['precipitations'][:]
    input_dt = hf['timestamps'][:]
    input_dt = np.array([datetime.datetime.strptime(x.decode('utf-8'), '%Y-%m-%d %H:%M:%S') for x in input_dt])


# pixel value range (0, 1)
max_rainfall_intensity = 60
input =  input / max_rainfall_intensity
# add batch and channel dimension to input. From [T, H, W] to [B, T, C, H, W]
X = np.expand_dims(input, axis=(0,2))
# create artificial output images
Y = np.zeros(X.shape)

# convert to tensor 
X = torch.tensor(X, dtype=torch.float32, device=device)
Y = torch.tensor(Y, dtype=torch.float32, device=device)

# predict
with torch.no_grad():   
    pred_Y = method._predict(X, Y)

# convert to numpy
pred_Y = pred_Y.cpu().numpy()
# reduce batch and channel dimension
pred_Y = np.squeeze(pred_Y, axis=(0,2))
# convert to orignal sacles
pred_Y = pred_Y * max_rainfall_intensity

output_dt = [input_dt[-1] + datetime.timedelta(minutes=30*(k+1)) for k in range(config['out_seq_length'])]
output_dt_str = [x.strftime('%Y-%m-%d %H:%M:%S') for x in output_dt]

# save results to h5py file
with h5py.File(output_h5_fname,'w') as hf:
    hf.create_dataset('precipitations', data=pred_Y)
    hf.create_dataset('timestamps', data=output_dt_str)



# from servir.visulizations.gif_creation import create_precipitation_gif

# wa_imerg_metadata = {'accutime': 30.0,
#     'cartesian_unit': 'degrees',
#     'institution': 'NOAA National Severe Storms Laboratory',
#     'projection': '+proj=longlat  +ellps=IAU76',
#     'threshold': 0.0125,
#     'timestamps': None,
#     'transform': None,
#     'unit': 'mm/h',
#     'x1': -21.4,
#     'x2': 30.4,
#     'xpixelsize': 0.04,
#     'y1': -2.9,
#     'y2': 33.1,
#     'yorigin': 'upper',
#     'ypixelsize': 0.04,
#     'zerovalue': 0}
# create_precipitation_gif(pred_Y, input_dt, 30, wa_imerg_metadata, '/home/cc/projects/nowcasting/temp/', 'test', gif_dur = 1000)