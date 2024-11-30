    
from servir.core.model_picker import ModelPicker
from pysteps.verification.probscores import CRPS
import numpy as np
import torch
import json
from servir.core.data_provider import IMERGDataModule
from pysteps.utils.spectral import rapsd
from pysteps.verification.detcatscores import det_cat_fct


# h5_dataset_location = "../data/events/5.h5"
h5_dataset_location = "temp/ghana_imerg_2011_2020_Oct.h5"
ir_h5_dataset_location = "temp/ghana_IR_2011_2020_oct.h5"
metadata_location = "../data/events/1/metadata.json"

thr = 10
get_crps = False

# model_type = 'steps'
# model_config_location = 'configs/gh_imerg/PySTEPS.py'
# model_save_location = None
# use_gpu = False


# model_type = 'linda'
# model_config_location = 'configs/gh_imerg/LINDA.py'
# model_save_location = None
# use_gpu = False

# model_type = 'dgmr'
# model_config_location = 'configs/gh_imerg/DGMR.py'
# model_save_location = 'temp/DGMR-epoch=39.ckpt'
# use_gpu = True

model_type = 'dgmr_ir'
model_config_location = 'configs/gh_imerg/DGMR.py'
model_save_location = 'temp/DGMR_IR-epoch=17.ckpt'
use_gpu = True


with open(metadata_location) as jsonfile:
    geodata_dict = json.loads(jsonfile.read())

dataset_type = 'ghana_monthly' # ghana, ghana_monthly and wa

data_provider =  IMERGDataModule(
        forecast_steps = 12,
        history_steps = 8,
        imerg_filename = h5_dataset_location,
        ir_filename = ir_h5_dataset_location,
        batch_size = 32,
        image_shape = (64,64),
        normalize_data=False,
        dataset = dataset_type)

data_loader = data_provider.test_dataloader()

crps_dict = {'dgmr':{},
             'dgmr_ir':{},
             'steps':{},
            }
psd_dict = {'dgmr':{},
             'dgmr_ir':{},
             'steps':{},
            }
csi_dict = {'dgmr':{},
             'dgmr_ir':{},
             'steps':{},
            }



for j in range(12):
    crps_dict['dgmr'][str(j)] = []
    crps_dict['dgmr_ir'][str(j)] = []
    crps_dict['steps'][str(j)] = []
    
    psd_dict['dgmr'][str(j)] = []
    psd_dict['dgmr_ir'][str(j)] = []
    psd_dict['steps'][str(j)] = []
    
    csi_dict['dgmr'][str(j)] = []
    csi_dict['dgmr_ir'][str(j)] = []
    csi_dict['steps'][str(j)] = []    
    



if get_crps:
    model_picker = ModelPicker(model_type, model_config_location, model_save_location, use_gpu)
    model_picker.load_model()

    for index, data_sample_batch in enumerate(data_loader):
        x, x_ir, y = data_sample_batch
        print("starting predictions for batch {}".format(index))
        if model_type in ['steps', 'lagrangian', 'naive', 'linda']:
            x = x.numpy()[:,:,0,:,:]
            y = y.numpy()[:,:,0,:,:]

            for data_sample_index in range(len(x)):
                try:
                    predicted_output = model_picker.predict(np.nan_to_num(x[data_sample_index]))
                    for j in range(12):
                        crps_dict[model_type][str(j)].append(CRPS(predicted_output[:,0:j,: ], y[data_sample_index][0:j]))
                except:
                    pass
        elif model_type in ['dgmr']:
            predicted_output = torch.tensor(model_picker.predict(x))
            rearanged_output = predicted_output.numpy().transpose(1, 0, 2, 3, 4, 5)[:,:,:,0,:,:]
            for data_sample_index in range(len(rearanged_output)):
                output = rearanged_output[data_sample_index]
                for j in range(12):
                    crps_dict[model_type][str(j)].append(CRPS(output[:,0:j,:,:], y.numpy()[:,:,0,:,:][data_sample_index][0:j,:,:]))
                
        elif model_type in ['dgmr_ir']:
            predicted_output = torch.tensor(model_picker.predict(x, x_ir))
            print(predicted_output.shape)

            rearanged_output = predicted_output.numpy().transpose(1, 0, 2, 3, 4, 5)[:,:,:,0,:,:]
            for data_sample_index in range(len(rearanged_output)):
                output = rearanged_output[data_sample_index]
                for j in range(12):
                    crps_dict[model_type][str(j)].append(CRPS(output[:,0:j,:,:], y.numpy()[:,:,0,:,:][data_sample_index][0:j,:,:]))
        

    np.save(model_type + '_crps.npy',crps_dict[model_type])

model_picker = ModelPicker(model_type, model_config_location,model_save_location, use_gpu)
model_picker.load_model(get_ensemble=False)
errored_out = 0

for index, data_sample_batch in enumerate(data_loader):
    x, x_ir, y = data_sample_batch
    print("starting predictions for batch {}".format(index))
    if model_type in ['steps', 'lagrangian', 'naive', 'linda']:
        x = x.numpy()[:,:,0,:,:]
        y = y.numpy()[:,:,0,:,:]

        for data_sample_index in range(len(x)):
            try:
                predicted_output = np.nan_to_num(model_picker.predict(x[data_sample_index]))
                for j in range(12):
                    csi_dict[model_type][str(j)].append(det_cat_fct(predicted_output[0,0:j,: ], y[data_sample_index][0:j], thr=thr)['CSI'])
                    psd_dict[model_type][str(j)].append(rapsd(predicted_output[0,j,:]))
            except:
                errored_out += 1
    elif model_type in ['dgmr']:
        predicted_output = torch.tensor(model_picker.predict(x))
        print(predicted_output.shape)

        rearanged_output = predicted_output.numpy().transpose(1, 0, 2, 3, 4, 5)[:,:,:,0,:,:]
        for data_sample_index in range(len(rearanged_output)):
            output = rearanged_output[data_sample_index]
            for j in range(12):
                csi_dict[model_type][str(j)].append(det_cat_fct(output[0][0:j,:,:], y.numpy()[:,:,0,:,:][data_sample_index][0:j,:,:], thr=thr))
                psd_dict[model_type][str(j)].append(rapsd(output[0][j]))
    elif model_type in ['dgmr_ir']:
        predicted_output = torch.tensor(model_picker.predict(x, x_ir))
        print(predicted_output.shape)

        rearanged_output = predicted_output.numpy().transpose(1, 0, 2, 3, 4, 5)[:,:,:,0,:,:]
        for data_sample_index in range(len(rearanged_output)):
            output = rearanged_output[data_sample_index]
            for j in range(12):
                csi_dict[model_type][str(j)].append(det_cat_fct(output[0][0:j,:,:], y.numpy()[:,:,0,:,:][data_sample_index][0:j,:,:], thr=thr))
                psd_dict[model_type][str(j)].append(rapsd(output[0][j]))
    

np.save(model_type + '_rapsd.npy',psd_dict[model_type])
np.save(model_type + '_' +str(thr) + '_csi.npy',csi_dict[model_type])
print("total errored out = ", errored_out)