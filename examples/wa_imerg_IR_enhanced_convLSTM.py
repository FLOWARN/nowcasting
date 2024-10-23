import os
import sys
base_path = "/home1/zhang2012/nowcasting/"#'/home/cc/projects/nowcasting' #
sys.path.append(base_path)

import h5py 
import time
import torch


from servir.core.distribution import get_dist_info
from nowcasting.servir.core.trainer_convlstm import train
from servir.datasets.dataLoader_wa_imerg_IR import waImergIRDataset, waImergIRDataset_withMeta
from servir.utils.config_utils import load_config
from servir.utils.logger_utils import logging_setup, logging_env_info, logging_config_info, logging_method_info

from servir.methods.ConvLSTM import ConvLSTM


method_name = 'ConvLSTM'
dataset_name = 'wa_imerg_IR'


# Results base path for logging, working dirs, etc. 
base_results_path = os.path.join(base_path, f'results/{dataset_name}')
if not os.path.exists(base_results_path):
    os.makedirs(base_results_path)

print(f'results path : {base_results_path}')

# logging setup
logging_setup(base_results_path, fname=f'{method_name}.log')   
print('logging file created')
# log env info
logging_env_info()
print('env info logged')

##=============Read In Configurations================##
# Load configuration file
config_path = os.path.join(base_path, f'configs/{dataset_name}', f'{method_name}.py') 

if os.path.isfile(config_path):
    print('config file found')
else:
    print(f'config file NOT found! config_path = {config_path}')

config = load_config(config_path)


# log config
logging_config_info(config)
print('configuration file logged')


##==================Data Loading=====================##
# where to load data
dataPath = os.path.join(base_path, 'data')

imerg_fPath = os.path.join(dataPath, 'wa_imerg/wa_imerg.h5')
IR_fPath = os.path.join(dataPath, 'wa_IR/wa_IR.h5')
pred_IR_fPath = os.path.join(base_path, 'results/wa_IR/IR_predictions_skip_loss.h5')

# training data from 2020-06-01 to 2020-08-18 
trainSet = waImergIRDataset(imerg_fPath, IR_fPath, pred_IR_fPath, '2020-06-01', '2020-08-18',\
                            config['in_seq_length'], config['out_seq_length'], config['pred_IR_length'])

# validation data from 2020-08-18 to 2020-08-25
valSet = waImergIRDataset(imerg_fPath, IR_fPath, pred_IR_fPath, '2020-08-18',  '2020-08-25',\
                        config['in_seq_length'], config['out_seq_length'], config['pred_IR_length'])

# testing data from 2020-08-25 to 2020-09-01, meta data is included for saving results
testSet = waImergIRDataset_withMeta(imerg_fPath, IR_fPath, pred_IR_fPath, '2020-08-25', '2020-09-01',\
                                config['in_seq_length'], config['out_seq_length'], config['pred_IR_length'])

print('Dataset created.')
print(f'training_len = {len(trainSet)}')
print(f'val_len = {len(valSet)}')
print(f'test_len = {len(testSet)}')

dataloader_train = torch.utils.data.DataLoader(trainSet, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
dataloader_val = torch.utils.data.DataLoader(valSet, batch_size=config['val_batch_size'], shuffle=True, pin_memory=True) 
dataloader_test = torch.utils.data.DataLoader(testSet, batch_size=config['val_batch_size'], shuffle=False, pin_memory=True)   

# update config
config['steps_per_epoch'] = len(dataloader_train)
##==================Setup Method=====================##
# get device
print(f'There are total {torch.cuda.device_count()} GPUs on current node')

if (config['use_gpu']) and torch.cuda.is_available(): 
    device = torch.device('cuda:0')
    gpu = torch.cuda.get_device_properties(device)
else:
    device = torch.device('cpu')

config['device'] = device

# setup method
method = ConvLSTM(config)

# # log method info
# logging_method_info(config, method, device)
# print('method setup')
#==============Distribution=========================##

# setup distribution
config['rank'], config['world_size'] = get_dist_info()

##==================Training=========================##
# # path and name of best model
para_dict_fpath = os.path.join(base_results_path, 'imerg_ir_fsss_params.pth')
print(f'model parameters saved at {para_dict_fpath}')

checkpoint_fname = os.path.join(base_results_path, 'imerg_ir_fsss_checkpoint.pth')
print(f'model training checkpoint saved at {para_dict_fpath}')

train(dataloader_train, dataloader_val, method, config, para_dict_fpath, checkpoint_fname)    
##==================Testing==========================## 

# # Loads best model’s parameter dictionary 
# method.model.load_state_dict(torch.load(para_dict_fpath))

# st = time.time()
# test_loss, test_pred, test_meta = method.test(dataloader_test, gather_pred = True)
# print(time.time()-st)
# # save results to h5py file
# with h5py.File(os.path.join(base_results_path, 'imerg_only_predictions.h5'),'w') as hf:
#     hf.create_dataset('precipitations', data=test_pred)
#     hf.create_dataset('timestamps', data=test_meta)

# print(f'results saved at {os.path.join(base_results_path, "imerg_only_predictions.h5")}')


            
    










