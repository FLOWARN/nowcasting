import os
import sys
base_path = '/home/cc/projects/nowcasting'
sys.path.append(base_path)

import h5py 
import torch
import datetime
import numpy as np
from matplotlib import pyplot as plt
from pysteps.utils import transformation
from pysteps.visualization import plot_precip_field

from servir.datasets.dataLoader_wa_imerg import load_wa_imerg_data_from_h5
from servir.datasets.dataLoader_wa_IR import load_IR_data_from_h5    

#================Specification=========================#
method_name = 'ConvLSTM'

dataset1_name = 'wa_imerg'
dataset2_name = 'wa_IR'

data1_fname = 'wa_imerg.h5'
data2_fname = 'wa_IR.h5'

# new data name
dataset_name = 'wa_IR'

st = '2020-08-25' 
ed = '2020-09-01'

imerg_normalize_method = None #'log-normal'
IR_normalize_method = None #'normal'



# test run on local machine
if base_path == '/home/cc/projects/nowcasting':
    data2_fname = 'wa_IR_08.h5'

##==================Data Loading=====================##
# where to load data
f1name = os.path.join(base_path, 'data', dataset1_name, data1_fname)
f2name = os.path.join(base_path, 'data', dataset2_name, data2_fname)

imergs, imerg_dts, imerg_mean, imerg_std, imerg_max, imerg_min = load_wa_imerg_data_from_h5(f1name,start_date= st, end_date=ed)
IRs, IR_dts, IR_mean, IR_std, IR_max, IR_min = load_IR_data_from_h5(f2name, start_date= st, end_date=ed)

# get the corresponding IRs for each imerg
IRs_ind = [list(IR_dts).index(t) for t in imerg_dts]
IRs = IRs[IRs_ind]
IR_dts = IR_dts[IRs_ind]


# look at the histogram of the data
plt.figure()
plt.hist(imergs.flatten(), bins=100)
plt.title('imergs histogram')

plt.figure()
plt.hist(IRs.flatten(), bins=100)
plt.title('IRs histogram')

# check the images of a sample
sample_dt = datetime.datetime(2020, 8, 25, 11, 0)

imerg_idx = list(imerg_dts).index(sample_dt)
IR_idx = list(IR_dts).index(sample_dt)

imerg = imergs[imerg_idx]
IR = IRs[IR_idx]    

plt.figure()
plt.imshow(imerg, cmap='gray')
plt.title(f'imerg - {datetime.datetime.strftime(sample_dt, "%Y-%m-%d %H:%M:%S")}')
plt.figure()
plot_precip_field(imerg) 
plt.title(f'imerg - {datetime.datetime.strftime(sample_dt, "%Y-%m-%d %H:%M:%S")}')


plt.figure()    
plt.imshow(IR, cmap='gray')
plt.title(f'IR - {datetime.datetime.strftime(sample_dt, "%Y-%m-%d %H:%M:%S")}')


# find a threshold for the IR data
IR_threshold = 240
f_IR = np.where(IR>IR_threshold, np.nan, -IR)
plt.figure()    
plt.imshow(f_IR, cmap='gray')


# compare the number of ignored pixels
imerg_ignore_num = np.sum(np.where(imergs==0, 1.0, 0.0))
IRs_ignore_num = np.sum(np.where(IRs<=IR_threshold, 0.0, 1.0))

print(f'number of zero value pixels in imergs: {imerg_ignore_num}')
print(f'number of >{IR_threshold} pixels in IRs: {IRs_ignore_num}')

# histogram of the imerg >0 and IRs>treshold
plt.figure()
plt.hist(imergs[imergs>0].flatten(), bins=100)
plt.title('imergs >0 histogram')

plt.figure()
plt.hist(IRs[IRs<=IR_threshold].flatten(), bins=100)
plt.title(f'IRs <={IR_threshold} histogram')

# Apply normal to IR data
# find the statistics of the thresholded IR data
IR_thresholded = np.where(IRs<=IR_threshold, IRs, np.nan)

IR_mean = np.nanmean(IR_thresholded)
IR_std = np.nanstd(IR_thresholded)
IR_max = np.nanmax(IR_thresholded)
IR_min = np.nanmin(IR_thresholded)

# if normalize, what is the min and max of the normalized data
IRs_norm_max = (IR_max - IR_mean) /IR_std 
IRs_norm_min = (IR_min - IR_mean) /IR_std 

print(f'IRs_norm_max: {IRs_norm_max}')
print(f'IRs_norm_min: {IRs_norm_min}')


# transform the IR data
replacevalue = -2.0
def IR_neg_scale(data, threshold, replacevalue, IR_max, IR_min):
    new = np.where(data>threshold, replacevalue, -(2*(data-IR_min)/(IR_max-IR_min)-1))
    return new

IRs_transformed = IR_neg_scale(IRs, IR_threshold, replacevalue, IR_max, IR_min)
plt.figure()
plt.hist(IRs_transformed[IRs_transformed>replacevalue].flatten(), bins=100)    


# Apply log-normal to imerg data
def imerg_log_normalize(data, threshold=0.1, zerovalue=-2.0):
    new = np.where(data < threshold, zerovalue, np.log10(data))
    return new

zerovalue = -2.0
imergs_transformed = imerg_log_normalize(imergs, threshold=0.1, zerovalue=zerovalue)
plt.figure()
plt.hist(imergs_transformed[imergs_transformed>zerovalue].flatten(), bins=100)    

imergs_recovered = np.where(imergs_transformed == zerovalue, 0.0, np.power(10, imergs_transformed))

plt.figure()
plot_precip_field(imergs_recovered[imerg_idx]) 
plt.title(f'imergs_recovered - {datetime.datetime.strftime(sample_dt, "%Y-%m-%d %H:%M:%S")}')
