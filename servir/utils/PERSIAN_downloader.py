# -*- coding: utf-8 -*-
import os
import requests
import gzip
import shutil
from datetime import datetime, timedelta
import urllib3
import subprocess

import rasterio
from rasterio.transform import from_origin
from rasterio.windows import from_bounds
import numpy as np
from servir.utils.m_tif2h5py import tif2h5py


def clipper(input_tiff_dir, final_output_dir, input_raw_data_dir,event_id = 'PDIR_data'):
    """
    Function to clip the GeoTIFF files to a specific bounding box and save them in a new directory.
    The function also removes the original folders after processing.
    The bounding box is defined by the coordinates:
    crop_xmin = -21.4
    crop_xmax = 30.4
    crop_ymin = -2.9
    crop_ymax = 33.1
    The nodata value is set to -9999 and the compression method is set to 'DEFLATE'.
    The function uses the rasterio library to read and write the GeoTIFF files.

    Args:
        input_tiff_dir (_type_): _description_
        final_output_dir (_type_): _description_
        input_raw_data_dir (_type_): _description_
        event_id (str, optional): _description_. Defaults to 'PDIR_data'.
    """
    crop_xmin = -21.4
    crop_xmax = 30.4
    crop_ymin = -2.9
    crop_ymax = 33.1

    nodata_value = -9999
    compression = 'DEFLATE'

    # Input directory containing the GeoTIFF files
    # input_tiff_dir = "pdir_tiff"
    # Output directory for cropped TIFFs
    # final_output_dir = "final_tiff"
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)

    # Loop through all TIFF files in the input directory
    for filename in os.listdir(input_tiff_dir):
        if filename.lower().endswith(".tif"):
            input_tiff_path = os.path.join(input_tiff_dir, filename)
            output_tiff_path = os.path.join(final_output_dir, filename)
            
            
            with rasterio.open(input_tiff_path) as src:
                
                window = from_bounds(crop_xmin, crop_ymin, crop_xmax, crop_ymax, src.transform)
                
                data = src.read(window=window)
                
                new_transform = rasterio.windows.transform(window, src.transform)
                
                
                profile = src.profile.copy()
                profile.update({
                    'height': window.height,
                    'width': window.width,
                    'transform': new_transform,
                    'nodata': nodata_value,
                    'compress': compression
                })
                
                with rasterio.open(output_tiff_path, 'w', **profile) as dst:
                    dst.write(data)
            
    print("Cropping complete for all TIFF files.")

    # Remove the folders pdirnow_data and pdir_tiff after processing
    folders_to_remove = [input_raw_data_dir, input_tiff_dir]
    for folder in folders_to_remove:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            

    print("------FILES CLIPPED----------")
    
    meta_fname = 'PDIR_metadata.json'
    
    tif2h5py(tif_directory=final_output_dir, 
             h5_fname= "/vol_efthymios/NFS07/en279/SERVIR/TITO_test3/ML/nowcasting/data/events/"+ str(event_id)+'.h5', 
             meta_fname=meta_fname, 
             x1=crop_xmin, y1=crop_ymin, 
             x2=crop_xmax, y2=crop_ymax)

    
    print("------h5 FILE PRODUCED----------")


def converter(input_dir, output_dir):
    """
    Function to convert binary files to GeoTIFF format.
    The function reads binary files from the input directory, processes them, and writes the output as GeoTIFF files in the output directory.
    The function uses the rasterio library to handle the GeoTIFF format.


    Args:
        input_dir (_type_): _description_
        output_dir (_type_): _description_
    """
    #Data Directory
    # input_dir = "pdirnow_data"  
    # output_dir = "pdir_tiff"    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    nrows = 3000
    ncols = 9000
    pixel_size = 0.04  
    x_min = -180.0
    y_max = 60.0
    transform = from_origin(x_min, y_max, pixel_size, pixel_size)
    nodata_raw = -9999    
    nodata_value = -99.99 
    for filename in os.listdir(input_dir):
        if filename.endswith(".bin"):
            bin_file = os.path.join(input_dir, filename)
            print(f"Processing {bin_file}...")
                
            try:
                raw_data = np.fromfile(bin_file, dtype=np.int16)
                raw_data = raw_data.reshape((nrows, ncols))
            except Exception as e:
                print(f"Error reading {bin_file}: {e}")
                continue
            
            
            data = raw_data.astype(np.float32)
            data[raw_data == nodata_raw] = np.nan
            data = data / 100.0
                
            roll_value = int(ncols / 2)  
            data_shifted = np.roll(data, shift=-roll_value, axis=1)
                
            data_shifted = np.where(np.isnan(data_shifted), nodata_value, data_shifted)
                    
            base_name = os.path.splitext(filename)[0]  
            out_filename = base_name + ".tif"
            out_file = os.path.join(output_dir, out_filename)
            
            
            with rasterio.open(
                out_file,
                'w',
                driver='GTiff',
                height=nrows,
                width=ncols,
                count=1,
                dtype='float32',
                crs='EPSG:4326',
                transform=transform,
                nodata=nodata_value,
                compress='DEFLATE'  
            ) as dst:
                dst.write(data_shifted, 1)
            
            print(f"Written GeoTIFF: {out_file}")

    print("_____CONVERTED TO TIFF______")


def download_files(input_date_str, save_dir , input_tiff_dir, output_tiff_dir_name, event_id):
    """
    Function to download PERSIAN data files from a given URL.
    The function constructs the URL based on the input date and downloads the files in a specified directory.
    It also decompresses the downloaded files and renames them according to a specific format.
    The function uses the requests library to handle HTTP requests and the gzip library to decompress the files.
    The function also calls the converter and clipper functions to process the downloaded files.
    The function uses the datetime library to handle date and time operations.
    The function uses the subprocess library to run external scripts.

    Args:
        input_date_str (_type_): _description_
        save_dir (_type_): _description_
        input_tiff_dir (_type_): _description_
        output_tiff_dir_name (_type_): _description_
        event_id (_type_): _description_
    """
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # -------------------------
    # Input Date Configuration
    # -------------------------

    # input_date_str = "2021-01-01:00" #Only thing to do
    input_dt = datetime.strptime(input_date_str, "%Y-%m-%d:%H")

    # Define the download interval:
    start_dt = input_dt - timedelta(hours=8)  # start date is  8 hours before
    end_dt   = input_dt - timedelta(hours=1)  # end 1 hour back from input date

    # --------------------------
    # Download Directory Setup
    # --------------------------
    # save_dir = "pdirnow_data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # --------------------------
    # Downloading Loop
    # --------------------------
    current_dt = start_dt
    while current_dt <= end_dt:
        # Build date components for URL and filename
        YYYY = current_dt.strftime("%Y")
        YY   = current_dt.strftime("%y")
        MM   = current_dt.strftime("%m")
        DD   = current_dt.strftime("%d")
        HH   = current_dt.strftime("%H")
        mm   = current_dt.strftime("%M")
        ss   = current_dt.strftime("%s")
        

        # Build filename and URL
        # Expected filename format: pdirnow1hYYMMDDHH.bin.gz
        filename_for_saving = f"pdirnow.pdir.{YYYY}{MM}{DD}{HH}{mm}.1h.bin.gz"
        filename = f"pdirnow1h{YY}{MM}{DD}{HH}.bin.gz"
        url = f"https://persiann.eng.uci.edu/CHRSdata/PDIRNow/PDIRNow1hourly/{YYYY}/{filename}"
        
        # Define local paths
        local_gz_path = os.path.join(save_dir, filename)
        local_bin_path = os.path.join(save_dir, filename[:-3])  
        
        print(f"Downloading {url} ...")
        try:
            response = requests.get(url, stream=True, verify=False)
            if response.status_code == 200:
                # Save the .gz file
                with open(local_gz_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"Download complete: {filename}")
                
                # Decompress to get the .bin file
                with gzip.open(local_gz_path, 'rb') as f_in:
                    with open(local_bin_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print(f"Decompressed file: {local_bin_path}")
                
                # Remove the original .gz file
                os.remove(local_gz_path)
                print(f"Removed compressed file: {local_gz_path}")
                
                # Rename the file
                os.rename(local_bin_path, os.path.join(save_dir, filename_for_saving[:-3]))
            else:
                print(f"Failed to download {url} (HTTP Status: {response.status_code})")
        except Exception as e:
            print(f"Error downloading {url}: {e}")
        
        # Increment one hour
        current_dt += timedelta(hours=1)

    print("---------DOWNLOADING FILES COMPLETE-----------.")

    # --------------------------
    # Execute Additional Processing Scripts
    # --------------------------
    # Run converter.py
    # print("Executing converter.py ...")
    # subprocess.run(["python", "converter.py"], check=True)
    # print("converter.py executed.")

    converter(input_dir = save_dir, output_dir = input_tiff_dir)

    # # Run clipper.py
    # print("Executing clipper.py ...")
    # subprocess.run(["python", "clipper.py"], check=True)
    # print("clipper.py executed.")

    clipper(input_tiff_dir = input_tiff_dir, final_output_dir = output_tiff_dir_name, input_raw_data_dir = save_dir, event_id=event_id)


    print("All processing complete. Folders have been removed.")


def main():
    from servir.utils.download_data import initialize_event_folders
    save_dir = "/vol_efthymios/NFS07/en279/SERVIR/TITO_test3/ML/nowcasting/data/events/"
    raw_data_name = "PDIR_data_raw"
    input_tiff_name = "PDIR_data_tiff"
    output_tiff_dir_name = "PDIR_data_processed"
        
    final_save_input_folder = initialize_event_folders(folder_name = save_dir,event_id = raw_data_name )
    final_save_input_tiff_folder = initialize_event_folders(folder_name = save_dir,event_id = input_tiff_name )
    
    final_save_output_folder = initialize_event_folders(folder_name = save_dir,event_id = output_tiff_dir_name )

    download_files(input_date_str = "2021-01-01:00", save_dir = final_save_input_folder, input_tiff_dir = final_save_input_tiff_folder,
                   output_tiff_dir_name = final_save_output_folder, event_id="PDIR_data")


if __name__ == "__main__":
    main()