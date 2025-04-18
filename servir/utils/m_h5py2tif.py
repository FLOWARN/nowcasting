
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


def h5py2tif(h5_fname, meta_fname, tif_directory, num_predictions, method):
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

    os.makedirs(tif_directory, exist_ok=True)
    
    # Load the predictions
    with h5py.File(h5_fname, 'r') as hf:
        for index in range(num_predictions):
            if num_predictions == 1:
                pred_imgs = hf['precipitations'][:]
                output_dts = hf['timestamps'][:]
            else:
                pred_imgs = hf[str(index) + 'precipitations'][:]
                output_dts = hf[str(index) + 'timestamps'][:]
            output_dts = np.array([datetime.datetime.strptime(x.decode('utf-8'), '%Y-%m-%d %H:%M:%S') for x in output_dts])

            if method == 'convlstm':
                pred_imgs = np.insert(pred_imgs, 0, 0, axis=2)
                pred_imgs = np.insert(pred_imgs, -1, 0, axis=2)

            

            for i in range(len(output_dts)):
                dt_str = output_dts[i].strftime('%Y%m%d%H%M')
                if num_predictions == 1:
                    gridOutName = os.path.join(tif_directory, f"imerg.qpf.{dt_str}.30minAccum.tif")
                else:
                    os.makedirs(tif_directory + str(index)+'/', exist_ok=True)
                    gridOutName = os.path.join(tif_directory+ str(index)+'/', f"imerg.qpf.{dt_str}.30minAccum.tif")
                WriteGrid(gridOutName, pred_imgs[i], nx, ny, gt, proj)
                


if __name__ == "__main__":
    # h5_fname =  sys.argv[1] 
    # meta_fname = sys.argv[2]
    # tif_directory = sys.argv[3]
    
    
    h5py2tif(h5_fname='/vol_efthymios/NFS07/en279/SERVIR/TITO_test3/ML/servir_nowcasting_examples/temp/output_imerg.h5', 
             meta_fname='/vol_efthymios/NFS07/en279/SERVIR/TITO_test3/ML/servir_nowcasting_examples/temp/imerg_geotiff_meta.json', 
             tif_directory='/vol_efthymios/NFS07/en279/SERVIR/TITO_test3/precip')