
import os
import sys
import datetime
import h5py
import json
import numpy as np
import osgeo.gdal as gdal





def h5py2tif(h5_fname, meta_fname, tif_directory, num_predictions, method, dataset = 'IMERG'):
    """
    Function to convert h5py files to geotiff files.
    The function reads in the h5py file and extracts the precipitation data and timestamps.
    It then writes out the precipitation data to a geotiff file using the metadata from the meta_fname file.
    The geotiff file is saved in the tif_directory with the method name as a subdirectory.
    The function also creates the tif_directory if it does not exist.


    Args:
        h5_fname (_type_): _description_
        meta_fname (_type_): _description_
        tif_directory (_type_): _description_
        num_predictions (_type_): _description_
        method (_type_): _description_
        dataset (str, optional): _description_. Defaults to 'IMERG'.
    """
    def get_EF5_geotiff_metadata(meta_fname):
        # Reads in the metadata file and extracts the georeference information
        with open(meta_fname, "r") as outfile:
            meta = json.load(outfile)
            
            nx = meta['nx']
            ny = meta['ny'] 
            gt = meta['gt'] 
            proj = meta['proj']

            return nx, ny, gt, proj

    def WriteGrid(gridOutName, dataOut, nx, ny, gt, proj):
        """
        Function to write out a GeoTIFF based on georeference information in RefInfo
        """
        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(gridOutName, nx, ny, 1, gdal.GDT_Float32, ['COMPRESS=DEFLATE'])
        dst_ds.SetGeoTransform(gt)
        dst_ds.SetProjection(proj)
        dataOut.shape = (-1, nx)
        dst_ds.GetRasterBand(1).WriteArray(dataOut, 0, 0)
        dst_ds.GetRasterBand(1).SetNoDataValue(-9999.0)
        dst_ds = None


    nx, ny, gt, proj = get_EF5_geotiff_metadata(meta_fname)

    os.makedirs(tif_directory + method, exist_ok=True)
    
    # Load the predictions
    with h5py.File(h5_fname, 'r') as hf:
        for index in range(num_predictions):
            if num_predictions == 1:
                pred_imgs = hf['precipitations'][:][0]
                output_dts = hf['timestamps'][:]
            else:
                pred_imgs = hf[str(index) + 'precipitations'][:]
                output_dts = hf[str(index) + 'timestamps'][:]
            output_dts = np.array([datetime.datetime.strptime(x.decode('utf-8'), '%Y-%m-%d %H:%M:%S') for x in output_dts])

            # if method == 'convlstm':
            #     pred_imgs = np.insert(pred_imgs, 0, 0, axis=2)
            #     pred_imgs = np.insert(pred_imgs, -1, 0, axis=2)

            
            if dataset == 'IMERG':
                filename_qualifier = 'imerg.qpf.'
                filename_end = '.30minAccum'
            elif dataset == 'PDIR':
                filename_qualifier = 'PDIR.qpf.'
                filename_end = '.60minAccum'
            for i in range(len(output_dts)):
                dt_str = output_dts[i].strftime('%Y%m%d%H%M')
                if num_predictions == 1:
                    gridOutName = os.path.join(tif_directory + method, f"{filename_qualifier}{dt_str}{filename_end}.tif")
                else:
                    os.makedirs(tif_directory + method +'/'+ str(index)+'/', exist_ok=True)
                    gridOutName = os.path.join(tif_directory + method +'/'+ str(index)+'/', f"{filename_qualifier}{dt_str}{filename_end}.tif")
                WriteGrid(gridOutName, pred_imgs[i], nx, ny, gt, proj)
                