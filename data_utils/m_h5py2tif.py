
import os
import sys
import datetime
import h5py
import json
import numpy as np
import osgeo.gdal as gdal


####

# This file is for project pipline only! 
# The function below is used to convert a h5py file to tiff images 
# in a specified directory

####

# h5_fname = '/home/cc/projects/nowcasting/temp/output_imerg.h5'
# meta_fname = '/home/cc/projects/nowcasting/temp/imerg_giotiff_meta.json'
# tif_directory = '/home/cc/projects/nowcasting/temp/'

h5_fname =  sys.argv[1] 
meta_fname = sys.argv[2]
tif_directory = sys.argv[3]

def get_EF5_geotiff_metadata(meta_fname):

    with open(meta_fname, "r") as outfile:
      meta = json.load(outfile)
    
    nx = meta['nx']
    ny = meta['ny'] 
    gt = meta['gt'] 
    proj = meta['proj'] 

    return nx, ny, gt, proj

def WriteGrid(gridOutName, dataOut, nx, ny, gt, proj):
    #Writes out a GeoTIFF based on georeference information in RefInfo
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(gridOutName, nx, ny, 1, gdal.GDT_Float32, ['COMPRESS=DEFLATE'])
    dst_ds.SetGeoTransform(gt)
    dst_ds.SetProjection(proj)
    dataOut.shape = (-1, nx)
    dst_ds.GetRasterBand(1).WriteArray(dataOut, 0, 0)
    dst_ds.GetRasterBand(1).SetNoDataValue(-9999.0)
    dst_ds = None


nx, ny, gt, proj = get_EF5_geotiff_metadata(meta_fname)

# Load the predictions
with h5py.File(h5_fname, 'r') as hf:
    pred_imgs = hf['precipitations'][:]
    output_dts = hf['timestamps'][:]
    output_dts = np.array([datetime.datetime.strptime(x.decode('utf-8'), '%Y-%m-%d %H:%M:%S') for x in output_dts])

pred_imgs = np.insert(pred_imgs, 0, 0, axis=2)
pred_imgs = np.insert(pred_imgs, -1, 0, axis=2)


for i in range(len(output_dts)):
    dt_str = output_dts[i].strftime('%Y%m%d%H%M')
    gridOutName = os.path.join(tif_directory, f"imerg_{dt_str}.tif")
    WriteGrid(gridOutName, pred_imgs[i], nx, ny, gt, proj)

