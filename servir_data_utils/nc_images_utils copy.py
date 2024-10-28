import os
import sys
base_path = '/home/cc/projects/nowcasting' #"/home1/zhang2012/nowcasting/"#
sys.path.append(base_path)
import glob
import datetime
import h5py
import glob

import numpy as np
import netCDF4 as nc
from scipy.interpolate import griddata



# Resample function
def resample_Tb(old_lat, old_lon, old_data, lat_R, lon_R):
    old_coordinates = np.array(np.meshgrid(old_lat, old_lon)).T.reshape(-1, 2)
    new_coordinates = np.array(np.meshgrid(lat_R, lon_R)).T.reshape(-1, 2)

    new_data = np.full((len(lat_R), len(lon_R), old_data.shape[2]), np.nan)
    
    for t in range(old_data.shape[2]):
        new_data[:, :, t] = griddata(old_coordinates, old_data[:, :, t].flatten(), new_coordinates, method='nearest').reshape(len(lat_R), len(lon_R))
        
    return new_data


def processing(IR_imgs, IR_dts, year):
    # flip images up-down
    IR_imgs = np.dstack([np.flipud(IR_imgs[k]) for k in range(IR_imgs.shape[0])]).transpose(2, 0, 1)

    # IR_dts = np.array([datetime.datetime.strptime(x.decode('utf-8'), '%Y-%m-%d %H:%M:%S') for x in IR_dts])

    # initialize the first datetime
    dt = datetime.datetime(year, 10, 1, 0, 0)
    while dt < datetime.datetime(year, 10, 31, 23, 45):
        for k, dt in enumerate(IR_dts):
            if k == 0:
                continue
            if IR_dts[k] == IR_dts[k-1]:
                print(f'duplicate value at index: {k}')
                IR_imgs = np.delete(IR_imgs, k, axis=0)
                IR_dts = np.delete(IR_dts, k)
                break

            if IR_dts[k] - IR_dts[k-1] != datetime.timedelta(minutes=15):

                print(f'missing value at index: {k}')
                print(f'img_dts[k]: {IR_dts[k]}')
                print(f'img_dts[k-1]: {IR_dts[k-1]}')
                IR_imgs = np.insert(IR_imgs, k, IR_imgs[k-1], axis=0)
                IR_dts = np.insert(IR_dts, k, IR_dts[k-1] + datetime.timedelta(minutes=15))
                break

    return IR_imgs, IR_dts



def nc2h5py(dataPath, start_date, end_date, fname='wa_nc.h5'):
##==================Data Loading=====================##
    # find all .nc files
    files = glob.glob(dataPath+'/*.nc')

    # load the first file to get the lat and lon
    nc_data_t = nc.Dataset(os.path.join(dataPath, files[0]),'r')
    lat_t= nc_data_t.variables['lat'][:]
    lon_t= nc_data_t.variables['lon'][:]
    lat_R= np.arange(4.85, 11.15, 0.1)
    lon_R = np.arange(-3.95, 2.35, 0.1)


    lambda_val = 10.8
    nu = 10000 / lambda_val
    c1 = 1.19104E-5
    c2 = 1.43877

    # get only the files that are within the date range
    st = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    et = datetime.datetime.strptime(end_date, '%Y-%m-%d') 

    requested_files = []
    test = []
    for file in files:
        f_dt = datetime.datetime.strptime(file.split('HRSEVIRI_')[1][:13], '%Y%m%dT%H%M')

        if f_dt < et and f_dt >= st:
            requested_files.append(file)
            test.append(f_dt)

    test.sort()

    file_size = len(requested_files)
    batch_size = 10
    batches = file_size // batch_size

    # start loading the data
    IR_imgs = [] 
    IR_dts = []
    for i in range(batches):

        print(f'batch: {i}')

        ch9 = np.full((len(lat_t), len(lon_t), batch_size), np.nan)
        Tb = np.full((len(lat_t), len(lon_t), batch_size), np.nan)

        for j in range(batch_size):
            index = i*batch_size+j
            file = requested_files[index]

            nc_data_t = nc.Dataset(file)
            date_str = file.split('HRSEVIRI_')[1][:13]  # Extract date string from file name
            date_obj = datetime.datetime.strptime(date_str, '%Y%m%dT%H%M')

            IR_dts.append(date_obj)

            ch9[:, :, j] = nc_data_t.variables['channel_9'][:]
            Tb[:, :, j] = c2 * nu / np.log(1 + (c1 * nu**3 /ch9[:, :, j]))

        Tb_R= resample_Tb(lat_t, lon_t, Tb, lat_R, lon_R)
        Tb_R = Tb_R.transpose(2, 0, 1)  # [S, H, W]

        IR_imgs.append(Tb_R)

    # get the remaining files
    remaining_files_size = file_size - batches*batch_size
    ch9 = np.full((len(lat_t), len(lon_t), remaining_files_size), np.nan)
    Tb = np.full((len(lat_t), len(lon_t), remaining_files_size), np.nan)

    for j in range(remaining_files_size):
        index = batches* batch_size+j 
        file = requested_files[index]
        nc_data_t = nc.Dataset(file)
        
        # get datetime object
        date_str = file.split('HRSEVIRI_')[1][:13]  # Extract date string from file name
        date_obj = datetime.datetime.strptime(date_str, '%Y%m%dT%H%M')
        IR_dts.append(date_obj)

        ch9[:, :, j] = nc_data_t.variables['channel_9'][:]
        Tb[:, :, j] = c2 * nu / np.log(1 + (c1 * nu**3 /ch9[:, :, j]))

    Tb_R= resample_Tb(lat_t, lon_t, Tb, lat_R, lon_R)
    Tb_R = Tb_R.transpose(2, 0, 1)  # [S, H, W]

    IR_imgs.append(Tb_R)

    #==================================================================================================#

    # convert to numpy array
    IR_imgs = np.concatenate(IR_imgs, axis=0)

    # sort the files by datetime
    IR_dts = np.array(IR_dts)
    sorted_index_array = np.argsort(IR_dts)
    sorted_timestamps = IR_dts[sorted_index_array]

    sorted_IR = IR_imgs[sorted_index_array]
    sorted_timestamps_str = [datetime.datetime.strftime(x, '%Y-%m-%d %H:%M:%S') for x in sorted_timestamps]

    with h5py.File(os.path.join(dataPath, fname), 'w') as hf:
        hf.create_dataset('IRs', data=sorted_IR)
        hf.create_dataset('timestamps', data=sorted_timestamps_str)
        hf.create_dataset('mean', data = sorted_IR.mean())
        hf.create_dataset('std', data = sorted_IR.std())
        hf.create_dataset('max', data = sorted_IR.max())
        hf.create_dataset('min', data = sorted_IR.min()) 

    year = sorted_timestamps[0].year

    IR, IR_dts = processing(sorted_IR, sorted_timestamps, year)
    
    assert IR.shape[0] == 2976, f'IR shape: {IR.shape}'
    assert len(IR_dts) == 2976, f'len(IR_dts): {len(IR_dts)}'

    print('IR shape: ', IR.shape)   
    # convert to list of strings for saving
    IR_dts_str = [x.strftime('%Y-%m-%d %H:%M:%S') for x in IR_dts]

    with h5py.File(os.path.join(dataPath, fname), 'w') as hf:
        hf.create_dataset('IRs', data=IR)
        hf.create_dataset('timestamps', data=IR_dts_str)
        hf.create_dataset('mean', data = IR.mean())
        hf.create_dataset('std', data = IR.std())
        hf.create_dataset('max', data = IR.max())
        hf.create_dataset('min', data = IR.min()) 




if __name__ == "__main__":

    dataPath = os.path.join(base_path, 'data', 'ghana_IR')

    with h5py.File(os.path.join(dataPath, 'ghana_IR_2011_2020_oct.h5'), 'r') as hf:
        imgs = hf['IRs'][:]
        img_dts = hf['timestamps'][:]

    print(f'imgs shape: {imgs.shape}')
    # print(f'mean = {np.nanmean(imgs)}')
    # print(f'std = {np.nanstd(imgs)}')
    # print(f'max = {np.nanmax(imgs)}')
    # print(f'min = {np.nanmin(imgs)}')

    # # fill nan values with the nearest value
    # stats = 0
    # for i in range(20000, imgs.shape[0]):
    #     if i % 1000 == 0:
    #         print(f'processing image: {i}')
    #     # for this image, fill in the nan value with nearest value

    #     if np.isnan(imgs[i]).all():
    #         print(f'all nan values at index: {i}')
    #         imgs[i] = imgs[i-1]
    #         continue

    #     for j in range(imgs[i].shape[0]):
    #         for k in range(imgs[i].shape[1]):
    #             if np.isnan(imgs[i, j, k]):
    #                 stats = stats+1
    #                 if ~np.isnan(imgs[i, j-1, k]):
    #                     imgs[i, j, k] = imgs[i, j-1, k]
    #                 elif ~np.isnan(imgs[i, j+1, k]):
    #                     imgs[i, j, k] = imgs[i, j+1, k]
    #                 elif ~np.isnan(imgs[i, j, k-1]):
    #                     imgs[i, j, k] = imgs[i, j, k-1]
    #                 elif ~np.isnan(imgs[i, j, k+1]):
    #                     imgs[i, j, k] = imgs[i, j, k+1]
    #                 else:
    #                     print(f'no value to fill at index: {i, j, k}')




    # print(f'number of nan values: {stats}')


    # with h5py.File(os.path.join(dataPath, 'ghana_IR_2011_2020_oct.h5'), 'w') as hf:
    #     hf.create_dataset('IRs', data=imgs)
    #     hf.create_dataset('timestamps', data=img_dts)
    #     hf.create_dataset('mean', data = imgs.mean())
    #     hf.create_dataset('std', data = imgs.std())
    #     hf.create_dataset('max', data = imgs.max())
    #     hf.create_dataset('min', data = imgs.min())








    # year = 2020
    # dataPath = os.path.join(base_path, 'data', 'ghana_IR', str(year))
    # fname = f'ghana_{year}_oct.h5'

    # start_date = f'{year}-10-01'
    # end_date = f'{year}-11-01'
    # nc2h5py(dataPath, start_date, end_date, fname=f'ghana_{year}_oct.h5')


    # dataPath = os.path.join(base_path, 'data', 'ghana_IR')

    # imgs_list = []
    # imgs_dts_list = []
    # for year in range(2011, 2021):

    #     fname = f'ghana_{year}_oct.h5'

    #     with h5py.File(os.path.join(dataPath,  fname), 'r') as hf:
    #         imgs = hf['IRs'][:]
    #         img_dts = hf['timestamps'][:]

    #     # convert to float32 precision
    #     # imgs.dtype = np.float32
    #     imgs_list.append(imgs)

    #     img_dts = np.array([datetime.datetime.strptime(x.decode('utf-8'), '%Y-%m-%d %H:%M:%S') for x in img_dts])
    #     imgs_dts_list = imgs_dts_list + img_dts.tolist()

    # imgs = np.concatenate(imgs_list, axis=0)
    # imgs_dts_str = [x.strftime('%Y-%m-%d %H:%M:%S') for x in imgs_dts_list]

    # with h5py.File(os.path.join(dataPath, 'ghana_IR_2011_2020_oct.h5'), 'w') as hf:
    #     hf.create_dataset('IRs', data=imgs)
    #     hf.create_dataset('timestamps', data=imgs_dts_str)
    #     hf.create_dataset('mean', data = np.nanmean(imgs))
    #     hf.create_dataset('std', data = np.nanstd(imgs))
    #     hf.create_dataset('max', data = np.nanmax(imgs))
    #     hf.create_dataset('min', data = np.nanmin(imgs))

    









    # dataPath = os.path.join(base_path, 'data', 'ghana_IR')
    # fname = f'ghana_{year}_oct.h5'

    # with h5py.File(os.path.join(dataPath,  fname), 'r') as hf:
    #     imgs = hf['IRs'][:]
    #     img_dts = hf['timestamps'][:]
    #     mean = imgs.mean()
    #     std = imgs.std()
    #     max = imgs.max()
    #     min = imgs.min()

    # print(f'imgs shape: {imgs.shape}')
    
    # img_dts = np.array([datetime.datetime.strptime(x.decode('utf-8'), '%Y-%m-%d %H:%M:%S') for x in img_dts])

    # # initialize the first datetime
    # dt = datetime.datetime(2016, 10, 1, 0, 0)
    # while dt < datetime.datetime(2016, 10, 31, 23, 45):
    #     for k, dt in enumerate(img_dts):
    #         if k == 0:
    #             continue
    #         if img_dts[k] == img_dts[k-1]:
    #             print(f'duplicate value at index: {k}')
    #             imgs = np.delete(imgs, k, axis=0)
    #             img_dts = np.delete(img_dts, k)
    #             break

    #         if img_dts[k] - img_dts[k-1] != datetime.timedelta(minutes=15):

    #             print(f'missing value at index: {k}')
    #             print(f'img_dts[k]: {img_dts[k]}')
    #             print(f'img_dts[k-1]: {img_dts[k-1]}')
    #             imgs = np.insert(imgs, k, imgs[k-1], axis=0)
    #             img_dts = np.insert(img_dts, k, img_dts[k-1] + datetime.timedelta(minutes=15))
    #             break

    # img_dts = [x.strftime('%Y-%m-%d %H:%M:%S') for x in img_dts]


    # print('stop for debugging')
        
    # # flip images up-down
    # imgs = np.dstack([np.flipud(imgs[k]) for k in range(imgs.shape[0])]).transpose(2, 0, 1)

    
    # with h5py.File(os.path.join(dataPath, fname), 'w') as hf:
    #     hf.create_dataset('IRs', data=imgs)
    #     hf.create_dataset('timestamps', data=img_dts)
    #     hf.create_dataset('mean', data =mean)
    #     hf.create_dataset('std', data = std)
    #     hf.create_dataset('max', data = max)
    #     hf.create_dataset('min', data = min) 

    # # cropping 2 columns from the left and right
    # imgs = imgs[:, :, 1:-1]


    # imgs_IR = np.concatenate([imgs_06, imgs_07, imgs_08], axis=0)
    # print(f'imgs_IR shape: {imgs_IR.shape}')
    # img_IR_dts = img_dts_06 + img_dts_07 + img_dts_08
    # print(f'len(img_IR_dts): {len(img_IR_dts)}')

    # print(f'img_IR_dts[0]: {img_IR_dts[0]}')
    # print(f'img_IR_dts[-1]: {img_IR_dts[-1]}')


    # with h5py.File(os.path.join(dataPath, 'wa_IR.h5'), 'w') as hf:
    #     hf.create_dataset('IRs', data=imgs_IR)
    #     hf.create_dataset('timestamps', data=img_IR_dts)
    #     hf.create_dataset('mean', data = imgs_IR.mean())
    #     hf.create_dataset('std', data = imgs_IR.std())

