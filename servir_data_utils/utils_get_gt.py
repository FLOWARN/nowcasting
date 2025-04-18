import os
import sys
base_path = '/home/cc/projects/nowcasting'#"/home1/zhang2012/nowcasting/"#
sys.path.append(base_path)

import h5py
import datetime
import numpy as np  
import pandas as pd
from matplotlib import pyplot as plt
import torch

from servir.datasets.dataLoader_wa_imerg import load_wa_imerg_data_from_h5

base_path = '/home/cc/projects/nowcasting' #"/home1/zhang2012/nowcasting/"#
sys.path.append(base_path)


method_name = 'ConvLSTM'
dataset_name = 'wa_imerg'

st = '2020-08-25' 
ed = '2020-09-01'


in_seq_length = 12
out_seq_length = 12 

# true imerg data path
dataPath1 = os.path.join(base_path, 'data', 'wa_imerg')
data1_fname = os.path.join(dataPath1, 'wa_imerg.h5')

imgs, img_dts, _, _, _, _ = load_wa_imerg_data_from_h5(data1_fname, start_date= st, end_date=ed)


# get true and naive images
trues = []
naives = []
meta = []
for st_ind_i in range(imgs.shape[0]-in_seq_length-out_seq_length):

    dts_i = img_dts[st_ind_i+in_seq_length:st_ind_i+in_seq_length+out_seq_length] 
    # convert to list of str
    dts_str_i = [x.strftime('%Y-%m-%d %H:%M:%S') for x in dts_i]
    # For each sample, get the true images
    true_imgs_i = imgs[st_ind_i+in_seq_length:st_ind_i+in_seq_length+out_seq_length] 

    naive_imgs_i = np.stack([imgs[st_ind_i+in_seq_length-1] for _ in range(out_seq_length)], axis=0)


    trues.append(true_imgs_i)
    naives.append(naive_imgs_i)
    meta.append(dts_str_i)

trues_array = np.stack(trues, axis=0)
naives_array = np.stack(naives, axis=0)

# load the predictions
base_results_path = os.path.join(base_path, f'results/{dataset_name}')
# save results to h5py file
with h5py.File(os.path.join(base_results_path, 'imerg_true.h5'),'w') as hf:
    hf.create_dataset('precipitations', data=trues_array)
    hf.create_dataset('timestamps', data=meta)

# save results to h5py file
with h5py.File(os.path.join(base_results_path, 'imerg_naive.h5'),'w') as hf:
    hf.create_dataset('precipitations', data=naives_array)
    hf.create_dataset('timestamps', data=meta)
