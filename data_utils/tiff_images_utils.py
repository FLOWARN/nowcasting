
import os
import glob
import datetime
import h5py
import shutil


import numpy as np
import pandas as pd

# import osgeo.gdal as gdal
# from osgeo.gdalconst import GA_ReadOnly
from matplotlib import pyplot as plt




def tiff2h5py(fPath, fname='wa_imerg.h5', start_date='2011-10-01', end_date='2020-10-01'):
    """Function to load IMERG tiff data from the associate event folder

    Args:
        data_location (str): string path to the location of the event data

    Returns:
        precipitation (np.array): np.array of precipitations (not sorted by time)
        times (np.array): np.array of date times that match 1:q with precipitation
    """
    precipitations = []
    times = []
    files = glob.glob(os.path.join(fPath, 'imerg*.tif'))

    if len(files)>0:
        for file in files:
            date_str = file.split("/")[-1].split('.')[1]
            year = date_str[0:4]
            month = date_str[4:6]
            day = date_str[6:8]
            hour = date_str[8:10]
            minute = date_str[10:12]
            dt = datetime.datetime.strptime(year + '-'+ month + '-' + day + ' '+ hour + ':' + minute, '%Y-%m-%d %H:%M')

            # years = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']
            # if year in years and month == '10':
            #     tiff_data = gdal.Open(file, GA_ReadOnly)
            #     imageArray = np.array(tiff_data.GetRasterBand(1).ReadAsArray())
            #     # get ghana region (64 by 64)
            #     ghana_region = imageArray[219:283, 174:238]

            #     times.append(dt)
            #     precipitations.append(ghana_region)

            if dt >= datetime.datetime.strptime(start_date, '%Y-%m-%d') and dt < datetime.datetime.strptime(end_date, '%Y-%m-%d'):
                tiff_data = gdal.Open(file, GA_ReadOnly)
                imageArray = np.array(tiff_data.GetRasterBand(1).ReadAsArray())

                times.append(dt)
                precipitations.append(imageArray)

        times = np.array(times)
        # images in tensor [T, H, W]
        precipitation = np.transpose(np.dstack(precipitations), (2, 0, 1))

        sorted_index_array = np.argsort(times)
        sorted_timestamps = times[sorted_index_array]
        sorted_precipitation = precipitation[sorted_index_array]

    else:
        sorted_precipitation = None
        sorted_timestamps = None


    sorted_timestamps_dt = [x.strftime('%Y-%m-%d %H:%M:%S') for x in sorted_timestamps]
    with h5py.File(os.path.join(fPath, fname), 'w') as hf:
        hf.create_dataset('precipitations', data=sorted_precipitation)
        hf.create_dataset('timestamps', data=sorted_timestamps_dt)
        hf.create_dataset('mean', data=np.mean(sorted_precipitation))
        hf.create_dataset('std', data=np.std(sorted_precipitation))
        hf.create_dataset('max', data=sorted_precipitation.max())
        hf.create_dataset('min', data=sorted_precipitation.min())


    return sorted_precipitation, sorted_timestamps
    

def save_as_npy(fPath, dPath):

    fPath = os.path.join(base_path, 'data/wa_imerg_tif/')
    dPath = os.path.join(base_path, 'data/wa_imerg/')

    files = glob.glob(os.path.join(fPath, 'imerg*.tif'))
    sorted_files = sorted(files)

    # part1 = sorted_files[:len(sorted_files)//2]
    # part2 = sorted_files[len(sorted_files)//2:]

    for k, file in enumerate(sorted_files):

        new_fname = file.split('/')[-1].rsplit('.',1)[0] + '.npy'
        new_f = os.path.join(dPath, new_fname)

        if k % 1000 == 0:
            print(f'Processing {k} of {len(sorted_files)}')

        tiff_data = gdal.Open(file, GA_ReadOnly)
        imageArray = np.array(tiff_data.GetRasterBand(1).ReadAsArray())

        with open(new_f, 'wb') as f:
            np.save(f, imageArray)

def get_stats(fPath):
    base_path = '/home/cc/projects/nowcasting' 

    fPath = os.path.join(base_path, 'data/wa_imerg_tif/')

    cur_mean = 0.0
    cur_max = 0.0

    files = glob.glob(os.path.join(fPath, 'imerg*.tif'))
    sorted_files = sorted(files)


    for k, file in enumerate(sorted_files):

        new_fname = file.split('/')[-1].rsplit('.',1)[0] + '.npy'

        if k % 1000 == 0:
            print(f'Processing {k} of {len(sorted_files)}')

        tiff_data = gdal.Open(file, GA_ReadOnly)
        imageArray = np.array(tiff_data.GetRasterBand(1).ReadAsArray())


        cur_mean += np.mean(imageArray)
        cur_max = max(cur_max, np.max(imageArray))

    all_mean = cur_mean/len(files)

    square_sum = 0.0
    
    # find the standard deviation
    for k, file in enumerate(sorted_files):

        if k % 1000 == 0:
            print(f'Processing {k} of {len(sorted_files)}')

        tiff_data = gdal.Open(file, GA_ReadOnly)
        imageArray = np.array(tiff_data.GetRasterBand(1).ReadAsArray())

        square_sum += np.sum((imageArray - all_mean)**2)

    std = np.sqrt(square_sum/(len(files)*360*518))


    print(f'All mean: {all_mean}, std: {std}, max: {cur_max}')

    return all_mean, std, cur_max

# ===============================================================================
# ===========================Load GeoTiff format data============================
# =============================================================================== 

def get_EF5_geotiff_metadata(fPath='/home/cc/projects/nowcasting/data/wa_imerg/imerg_giotiff_meta.json'):

    with open(fPath, "r") as outfile:
      meta = json.load(outfile)
    
    nx = meta['nx']
    ny = meta['ny'] 
    gt = meta['gt'] 
    proj = meta['proj'] 

    return nx, ny, gt, proj

def WriteGrid(gridOutName, dataOut, nx, ny, gt, proj):
    import osgeo.gdal as gdal
    from osgeo.gdalconst import GA_ReadOnly
    #Writes out a GeoTIFF based on georeference information in RefInfo
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(gridOutName, nx, ny, 1, gdal.GDT_Float32, ['COMPRESS=DEFLATE'])
    dst_ds.SetGeoTransform(gt)
    dst_ds.SetProjection(proj)
    dataOut.shape = (-1, nx)
    dst_ds.GetRasterBand(1).WriteArray(dataOut, 0, 0)
    dst_ds.GetRasterBand(1).SetNoDataValue(-9999.0)
    dst_ds = None


def write_forcasts_to_geotiff(output_fPath, output_meta_fPath, resultsPath, model_config):
    nx, ny, gt, proj = get_EF5_geotiff_metadata()

    output_meta = pd.read_csv(output_meta_fPath)    
    with h5py.File(output_fPath,'r') as hf:
        output = hf['forcasts'][:]


    for i in range(output.shape[0]):
        i_meta = output_meta.iloc[i]
        # event_name = i_meta['event_name']
        out_dt = [datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S') for dt_str in i_meta['out_datetimes'].split(',')]

        method_path = os.path.join(resultsPath, f"{model_config['method']}")
        if not os.path.exists(method_path):
            os.mkdir(method_path)
        

        sample_results_path = os.path.join(method_path, str(i))
        if not os.path.exists(sample_results_path):
            os.mkdir(sample_results_path)
        
        precipitations = output[i, :, :, :] 
        for t in range(precipitations.shape[2]):
            precip_t = precipitations[:, :, t]
            gridOutName = os.path.join(sample_results_path, f"{out_dt[t].strftime('%Y%m%d%H%M')}.tif")
            WriteGrid(gridOutName, precip_t, nx, ny, gt, proj)


if __name__ == "__main__":

    base_path =  '/home1/zhang2012/nowcasting' #'/home/cc/projects/nowcasting' #

    fPath = os.path.join(base_path, 'data/wa_imerg/')


    files = glob.glob(os.path.join(fPath, 'imerg*.npy'))

    start_dt =  datetime.datetime.strptime('2017-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    end_dt = datetime.datetime.strptime('2023-06-01 00:00:00', '%Y-%m-%d %H:%M:%S')

    time_delta = datetime.timedelta(minutes=30) # this is fixed for IMERG data

    # get all datetimes between start and end with 30 minutes interval
    all_datetimes = [start_dt + i*time_delta for i in range(int((end_dt-start_dt)/time_delta))]

    # get all files coreesponding to the datetimes
    all_files = [os.path.join(fPath, f'imerg.{x.strftime("%Y%m%d%H%M")}.30minAccum.npy') for x in all_datetimes]
    
    count = 0
    for k, file in enumerate(all_files):

        # if k % 1000 == 0:
        #     print(f'Processing {k} of {len(all_files)}')

        if not os.path.exists(file):
            print(f'File {file} does not exist')
            count += 1
            # make a copy of the previous file and save it as the current file
            shutil.copy(all_files[k-1], file)




