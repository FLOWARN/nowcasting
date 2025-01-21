import json
from servir.core.model_picker import ModelPicker
from pysteps.verification.probscores import CRPS
import numpy as np
import json
from pysteps.utils.spectral import rapsd
from pysteps.verification.detcatscores import det_cat_fct
import torch
import matplotlib.pyplot as plt
import matplotlib


def generate_outputs(data_loader, model_type, model_config_location, model_save_location, use_gpu, produce_ensemble_outputs=False, event_name = None, model_output_location = 'results'):
    
    predicted_outputs = []
    
    if produce_ensemble_outputs:
        model_picker = ModelPicker(model_type, model_config_location, model_save_location, use_gpu)
        model_picker.load_model()

        for index, data_sample_batch in enumerate(data_loader):
            x, y = data_sample_batch
            print("starting predictions for batch {}".format(index))
            if model_type in ['steps', 'linda']:
                x = x.numpy()[:,:,0,:,:]
                y = y.numpy()[:,:,0,:,:]

                for data_sample_index in range(len(x)):
                    try:
                        predicted_output = model_picker.predict(np.nan_to_num(x[data_sample_index]))
                        predicted_outputs.append(predicted_outputs)
                    except:
                        pass
            elif model_type in ['dgmr']:
                predicted_output = torch.tensor(model_picker.predict(x))
                rearanged_output = predicted_output.numpy().transpose(1, 0, 2, 3, 4, 5)[:,:,:,0,:,:]
                for data_sample_index in range(len(rearanged_output)):
                    output = rearanged_output[data_sample_index]
                    predicted_outputs.append(output)
                     
            elif model_type in ['dgmr_ir']:
                predicted_output = torch.tensor(model_picker.predict(x, x_ir))
                print(predicted_output.shape)

                rearanged_output = predicted_output.numpy().transpose(1, 0, 2, 3, 4, 5)[:,:,:,0,:,:]
                for data_sample_index in range(len(rearanged_output)):
                    output = rearanged_output[data_sample_index]
                    predicted_outputs.append(output)
            else:
                raise Exception("The model cannot produce ensembles ")
                    
                
    model_picker = ModelPicker(model_type, model_config_location,model_save_location, use_gpu)
    model_picker.load_model(get_ensemble=False)
    errored_out = 0

    for index, data_sample_batch in enumerate(data_loader):
        x, y = data_sample_batch
        print("starting predictions for batch {}".format(index))
        if model_type in ['steps', 'lagrangian', 'naive', 'linda']:
            x = x.numpy()[:,:,0,:,:]
            y = y.numpy()[:,:,0,:,:]

            for data_sample_index in range(len(x)):
                predicted_output = np.nan_to_num(model_picker.predict(x[data_sample_index]))
                predicted_outputs.append(predicted_output)
                
        elif model_type in ['dgmr']:
            predicted_output = torch.tensor(model_picker.predict(x))

            rearanged_output = predicted_output.numpy().transpose(1, 0, 2, 3, 4, 5)[:,:,:,0,:,:]
            for data_sample_index in range(len(rearanged_output)):
                output = rearanged_output[data_sample_index]
                predicted_outputs.append(output)
                  
        elif model_type in [ 'convlstm']:
            # predicted_output = torch.tensor(model_picker.predict(x[:,:,0,:,:]))
            x = x.numpy()[:,:,0,:,:]
            y = y.numpy()[:,:,0,:,:]

            for data_sample_index in range(len(x)):
                print(data_sample_index)
                predicted_output = torch.tensor(model_picker.predict(x[data_sample_index])).numpy()
                predicted_outputs.append(predicted_output)
                        
              
        elif model_type in ['dgmr_ir']:
            predicted_output = torch.tensor(model_picker.predict(x, x_ir))

            rearanged_output = predicted_output.numpy().transpose(1, 0, 2, 3, 4, 5)[:,:,:,0,:,:]
            for data_sample_index in range(len(rearanged_output)):
                output = rearanged_output[data_sample_index]
                predicted_outputs.append(output)
        else:
            raise Exception("The model has not been implemented")
                
    predicted_outputs = np.array(predicted_outputs)
    np.save(model_output_location + model_type + '_'+str(event_name)+'_outputs.npy',predicted_outputs)
    return predicted_outputs
    

def evaluation(data_loader, thr_list, model_type, model_config_location, model_save_location, use_gpu, use_ensemble=False, event_name = None, model_output_location='results/'):

    crps_dict = {model_type:{},
                 'gt':{}
                }

    psd_dict = {model_type:{},
                 'gt':{}
                }

    
    csi_dict = {model_type:{},
                 'gt':{}
                }
    
    csi_dict_list = [csi_dict.copy() for x in thr_list]

    for j in range(12):
        crps_dict[model_type][str(j)] = []
        
        psd_dict[model_type][str(j)] = []
        psd_dict['gt'][str(j)] = []
        
    for x in range(len(thr_list)):
        for j in range(12):
            csi_dict_list[x][model_type][str(j)] = []
        
    if use_ensemble:
        model_picker = ModelPicker(model_type, model_config_location, model_save_location, use_gpu)
        model_picker.load_model()

        for index, data_sample_batch in enumerate(data_loader):
            x, y = data_sample_batch
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
        np.save(model_output_location+model_type +'_'+event_name+ '_crps.npy',crps_dict[model_type])

    model_picker = ModelPicker(model_type, model_config_location,model_save_location, use_gpu)
    model_picker.load_model(get_ensemble=False)
    errored_out = 0

    for index, data_sample_batch in enumerate(data_loader):
        x, y = data_sample_batch
        print("starting predictions for batch {}".format(index))
        if model_type in ['steps', 'lagrangian', 'naive', 'linda']:
            x = x.numpy()[:,:,0,:,:]
            y = y.numpy()[:,:,0,:,:]

            for data_sample_index in range(len(x)):
                # try:
                predicted_output = np.nan_to_num(model_picker.predict(x[data_sample_index]))
                print(predicted_output.shape)
                for j in range(12):
                    for index, thr in enumerate(thr_list): 
                        csi_dict_list[index][model_type][str(j)].append(det_cat_fct(predicted_output[0:j,:, :], y[data_sample_index][0:j], thr=thr)['CSI'])
                    psd_dict[model_type][str(j)].append(rapsd(predicted_output[j,:, :], return_freq=True, fft_method = np.fft))
                    psd_dict['gt'][str(j)].append(rapsd(y[data_sample_index][j, :, :], return_freq=True, fft_method = np.fft))
                        
                # except:
                #     errored_out += 1
        elif model_type in ['dgmr']:
            predicted_output = torch.tensor(model_picker.predict(x))

            rearanged_output = predicted_output.numpy().transpose(1, 0, 2, 3, 4, 5)[:,:,:,0,:,:]
            for data_sample_index in range(len(rearanged_output)):
                output = rearanged_output[data_sample_index]
                for j in range(12):
                    for index, thr in enumerate(thr_list): 
                        csi_dict_list[index][model_type][str(j)].append(det_cat_fct(output[0][0:j,:,:], y.numpy()[:,:,0,:,:][data_sample_index][0:j,:,:], thr=thr))
                    psd_dict[model_type][str(j)].append(rapsd(output[0][j],return_freq=True, fft_method = np.fft))
                    psd_dict['gt'][str(j)].append(rapsd(y.numpy()[:,:,0,:,:][data_sample_index][j,:,:],return_freq=True, fft_method = np.fft))
        
        
        elif model_type in [ 'convlstm']:
            # predicted_output = torch.tensor(model_picker.predict(x[:,:,0,:,:]))
            x = x.numpy()[:,:,0,:,:]
            y = y.numpy()[:,:,0,:,:]

            for data_sample_index in range(len(x)):
                print(data_sample_index)
                predicted_output = torch.tensor(model_picker.predict(x[data_sample_index])).numpy()
                print(predicted_output[j,:,:].shape)
                for j in range(12):
                        for index, thr in enumerate(thr_list): 
                            csi_dict_list[index][model_type][str(j)].append(det_cat_fct(predicted_output[:j,:,:], y[data_sample_index][0:j], thr=thr)['CSI'])
                        psd_dict[model_type][str(j)].append(rapsd(predicted_output[j,:,:], return_freq=True, fft_method = np.fft))
                        psd_dict['gt'][str(j)].append(rapsd(y[data_sample_index][j], return_freq=True, fft_method = np.fft))
                  
            # rearanged_output = predicted_output.numpy().transpose(1, 0, 2, 3, 4, 5)[:,:,:,0,:,:]
            # for data_sample_index in range(len(rearanged_output)):
            #     output = rearanged_output[data_sample_index]
            #     for j in range(12):
            #         csi_dict[model_type][str(j)].append(det_cat_fct(output[0][0:j,:,:], y.numpy()[:,:,0,:,:][data_sample_index][0:j,:,:], thr=thr))
            #         psd_dict[model_type][str(j)].append(rapsd(output[0][j],return_freq=True, fft_method = np.fft))
            #         psd_dict['gt'][str(j)].append(rapsd(y.numpy()[:,:,0,:,:][data_sample_index][j,:,:],return_freq=True, fft_method = np.fft))
                   
        elif model_type in ['dgmr_ir']:
            predicted_output = torch.tensor(model_picker.predict(x, x_ir))

            rearanged_output = predicted_output.numpy().transpose(1, 0, 2, 3, 4, 5)[:,:,:,0,:,:]
            for data_sample_index in range(len(rearanged_output)):
                output = rearanged_output[data_sample_index]
                for j in range(12):
                    for index, thr in enumerate(thr_list): 
                        csi_dict_list[index][model_type][str(j)].append(det_cat_fct(output[0][0:j,:,:], y.numpy()[:,:,0,:,:][data_sample_index][0:j,:,:], thr=thr))
                    psd_dict[model_type][str(j)].append(rapsd(output[0][j], return_freq=True, fft_method = np.fft))
                    psd_dict['gt'][str(j)].append(rapsd(y.numpy()[:,:,0,:,:][data_sample_index][j,:,:], return_freq=True, fft_method = np.fft))


    np.save(model_output_location + model_type + '_'+str(event_name)+'_rapsd.npy',psd_dict[model_type])
    np.save(model_output_location+'gt_'+str(event_name)+'_rapsd.npy',psd_dict['gt'])
    for index, thr in enumerate(thr_list):
        np.save(model_output_location+model_type + '_' +str(thr) + '_'+str(event_name)+ '_csi.npy',csi_dict_list[index][model_type])
    print("total errored out = ", errored_out)
    
    return crps_dict, psd_dict, csi_dict


def csi_boxplot(csi_val_dict, csi_model_names, thr):
    

    fig, ax = plt.subplots(figsize=(20,10)) 
    
    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)
        
        
    num_boxplots = len(csi_val_dict)
    color_options = ['#D7191C', '#2C7BB6', '#008000', '#009550']
    
    
    for i in range(1, 12):
        
        for j in range(num_boxplots):
            bp_data = np.array([x for x in csi_val_dict[j][str(i)]])
            bp_data = bp_data[~np.isnan(bp_data)]
            bp_1 = plt.boxplot([bp_data],widths = 0.9,  positions=[j+i*6-0.5], showmeans=True,showfliers=False, 
                                    patch_artist=True, meanprops={"markerfacecolor":"#D7191C", 
                                                                    "markeredgecolor": "black",
                                                                    "markersize": "15"})
            set_box_color(bp_1, color_options[j%(len(color_options))])

        
        ax.axvline(0+i*6 - 0.5 - 0.55, ls='--')
        ax.axvline((num_boxplots-1)+i*6 - 0.5 + 0.55, ls='--')
        
        
       
    plt.xticks(range(1, 12 * 6 + 1, 6),['','-3H','-2.5H','-2H','-1.5H','-1H','-0.5H','0','0.5H','1H','1.5H','2H'])

    for i in range(num_boxplots):
        plt.plot([], c=color_options[i%(len(color_options))], label=csi_model_names[i])
    
    plt.xlabel("Forecast Horizon (hours)")
    plt.ylabel("CSI ")
    plt.title("Critical Success Index at {} mm/h".format(thr))
    plt.legend()
    plt.savefig('results/csi_{}.png'.format(thr), dpi=100)
    


def rapsd_boxplot(rapsd_val_dict, rapsd_model_names, s=8):
    
    def plot_psd(psd_tuple):
        psd , freq = psd_tuple
        wvlength = s/freq 
        psd[psd==0] = 0.000001
        
        return np.log(psd), wvlength
    
    num_boxplots = len(rapsd_val_dict)
    
    for k in range(num_boxplots):
        psd_list = []
        for i in range(12):
            for j in range(len(rapsd_val_dict[k][str(i)])):
                psd, wvlength = plot_psd(rapsd_val_dict[k][str(i)][j])
                psd_list.append(psd)
                
        plt.plot(wvlength,np.mean(psd_list, axis=0), label=rapsd_model_names[k])

    plt.gca().invert_xaxis()
    ax1 = plt.gca()
    plt.xscale('log')
    ax1.set_xticks([20, 50, 100, 200, 300 ,500])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.legend()
    plt.xlabel('Wavelength (km)')
    plt.ylabel('Mean Radially Averaged Power Spectral Density')
    plt.savefig('results/PSD.png', dpi=300)

