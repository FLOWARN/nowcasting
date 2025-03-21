
from shutil import rmtree, copy
import os
from os import makedirs, listdir, rename, remove
import glob
from datetime import datetime as dt
from datetime import timedelta
import errno
import datetime
import time
import numpy as np
import re
import subprocess
import threading
import sys
import socket
import shutil
import smtplib
import json
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from multiprocessing.pool import ThreadPool
import requests
from bs4 import BeautifulSoup
import datetime as DT
import osgeo.gdal as gdal
from osgeo.gdal import gdalconst
from osgeo.gdalconst import GA_ReadOnly
import matplotlib.pyplot as plt
import time
import pandas as pd


def downloadIRData(start_date, end_date):
    pass
    # make sure that eumdac and datatailor are installed in the linux environment

    # eumdac download -c EO:EUM:DAT:MSG:HRSEVIRI -s 2020-07-01T00:00
    # -e 2020-08-01T00:00 --tailor "product: HRSEVIRI, format: netcdf4, projection: geographic, roi: {NSWE: [33.5, -3, -21.5, 30.5]}, 
    # filter:{ bands : [channel_9]}" -o D:\IR_nc\2020-01
    
    # command = 'eumdac download -c EO:EUM:DAT:MSG:HRSEVIRI -s '+ start_date + ' -e ' + end_date + '--tailor "product: HRSEVIRI, format: netcdf4, projection: geographic, roi: {NSWE: [33.5, -3, -21.5, 30.5]},filter:{ bands : [channel_9]}" -o D:\IR_nc\'
    


def ReadandWarp(gridFile,xmin,ymin,xmax,ymax, req_height, req_width, morph_size):

    #Read grid and warp to domain grid
    #Assumes no reprojection is necessary, and EPSG:4326
    rawGridIn = gdal.Open(gridFile, GA_ReadOnly)

    # Adjust grid
    pre_ds = gdal.Translate('OutTemp.tif', rawGridIn, options="-co COMPRESS=Deflate -a_nodata 29999 -a_ullr -180.0 90.0 180.0 -90.0")

    gt = pre_ds.GetGeoTransform()
    proj = pre_ds.GetProjection()
    nx = pre_ds.GetRasterBand(1).XSize
    ny = pre_ds.GetRasterBand(1).YSize
    NoData = 29999
    
    if morph_size:
        pixel_size_x = (xmax - xmin)/req_width
        pixel_size_y = (ymax - ymin)/req_height
    else:
        pixel_size_x = gt[1]
        pixel_size_y = gt[1]
    # pixel_size_x, pixel_size_y = getRequiredPixelSizes(nx, ny, (64,64))
    # Warp to model resolution and domain extents
    ds = gdal.Warp('', pre_ds, srcNodata=NoData, srcSRS='EPSG:4326', dstSRS='EPSG:4326', dstNodata='-9999', format='VRT', xRes=pixel_size_x, yRes=-pixel_size_y, outputBounds=(xmin,ymin,xmax,ymax))

    WarpedGrid = ds.ReadAsArray()
    new_gt = ds.GetGeoTransform()
    new_proj = ds.GetProjection()
    new_nx = ds.GetRasterBand(1).XSize
    new_ny = ds.GetRasterBand(1).YSize

    return WarpedGrid, new_nx, new_ny, new_gt, new_proj, {'projection':new_proj, 'x1':xmin , 'x2':xmax, 'y1':ymin, 'y2':ymax, 'yorigin':'upper'}

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

def getRequiredPixelSizes(nx, ny, requiredSize):
    x,y = requiredSize
    x_pixel_size = x/nx
    y_pixel_size = y/ny
    return x_pixel_size, y_pixel_size
    
    

def processIMERG(local_filename,llx,lly,urx,ury, req_height, req_width, morph_size=False):
    # Process grid
    # Read and subset grid
    NewGrid, nx, ny, gt, proj, metadata = ReadandWarp(local_filename,llx,lly,urx,ury, req_height, req_width, morph_size)
    # Scale value
    NewGrid = NewGrid*0.1
    return NewGrid, nx, ny, gt, proj, metadata


def get_file(filename,server,email):
   ''' Get the given file from jsimpsonhttps using curl. '''
   url = server + '/' + filename
   cmd = 'curl -sO -u ' + email + ':' + email + ' ' + url
   args = cmd.split()
   process = subprocess.Popen(args, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
   process.wait() # wait so this program doesn't end before getting all files#

def get_gpm_files(precipFolder,initial_timestamp, final_timestamp, ppt_server_path ,email, req_height, req_width):
    #path server
    server = ppt_server_path
    file_prefix = '3B-HHR-E.MS.MRG.3IMERG.'
    file_suffix = '.V07B.30min.tif'
    
    print(type(final_timestamp))
    final_date = final_timestamp + DT.timedelta(minutes=30)
    delta_time = datetime.timedelta(minutes=30)
    
    # Loop through dates
    current_date = initial_timestamp
    #acumulador_30M = 0
    
    while (current_date < final_date):
        initial_time_stmp = current_date.strftime('%Y%m%d-S%H%M%S')
        final_time = current_date + DT.timedelta(minutes=29)
        final_time_stmp = final_time.strftime('E%H%M59')
        final_time_gridout = current_date + DT.timedelta(minutes=30)
        folder = current_date.strftime('%Y/%m/')
        
        # #finding accum
        hours = (current_date.hour)
        minutes = (current_date.minute)
    
        # # Calculate the number of minutes since the beginning of the day.
        total_minutes = hours * 60 + minutes
    
        date_stamp = initial_time_stmp + '-' + final_time_stmp + '.' + f"{total_minutes:04}"

        filename = folder + file_prefix + date_stamp + file_suffix

        print('    Downloading ' + final_time_gridout.strftime('%Y-%m-%d %H:%M'))
        
        try:
            gridOutName = precipFolder+'imerg.qpe.' + final_time_gridout.strftime('%Y%m%d%H%M') + '.30minAccum.tif'
                
            # Download from NASA server
            if not os.path.isfile(gridOutName):
                get_file(filename,server,email)
                # Process file for domain and to fit EF5
                # Filename has final datestamp as it represents the accumulation upto that point in time
                local_filename = file_prefix + date_stamp + file_suffix
                
                # NewGrid_ghana, nx_ghana, ny_ghana, gt_ghana, proj_ghana, metadata_ghana = processIMERG(local_filename,xmin,ymin,xmax,ymax, req_height, req_width, morph_size = True)

                NewGrid_wa, nx_wa, ny_wa, gt_wa, proj_wa, metadata_wa = processIMERG(local_filename,xmin,ymin,xmax,ymax, req_height, req_width, morph_size = True)

                # NewGrid_ghana = NewGrid_wa[219:283, 174:238] for ghana from west africa -- no longer used since the metadata file generated is not accurate
                
                filerm = file_prefix + date_stamp + file_suffix
                
                # Write out processed filename
                # WriteGrid(gridOutName, NewGrid_ghana, nx_ghana, ny_ghana, gt_ghana, proj_ghana)
                WriteGrid(gridOutName, NewGrid_wa, nx_wa, ny_wa, gt_wa, proj_wa)

                os.remove(filerm)
            else:
                print("File already exists, continuing")
        except Exception as e:
            print(e)
            print(filename)
            pass

        # Advance in time
        current_date = current_date + delta_time
    
def month_converter(month):
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    return months.index(month) + 1

def get_event(csv_filename, event_id=1, start_hour =0, start_minute=0, end_hour=0, end_minute=0):
    events_df = pd.read_csv(csv_filename)
    # events_df['Event  Recorded'] = pd.to_datetime(events_df['Event  Recorded'], format='mixed')
    date_string = events_df['Event  Recorded Date'].values[0]
    month = date_string.split(' ')[0]
    day = int(date_string.split(' ')[1][:-1])
    year = int(date_string.split(' ')[-1])
    # event_date = datetime.datetime(year, month_converter(month), day, 0, 0, 0)
    start_date = datetime.datetime(year, month_converter(month), day, start_hour, start_minute, 0)
    
    # end_date = event_date + timedelta(days=1)
    end_date = datetime.datetime(year, month_converter(month), day, end_hour, end_minute, 0)
    return start_date, end_date


def get_event_for_forecasting(start_date, end_date, event_label=1, start_hour =0, start_minute=0, end_hour=0, end_minute=0):
    # events_df['Event  Recorded'] = pd.to_datetime(events_df['Event  Recorded'], format='mixed')
    start_date_string = start_date
    start_month = start_date.month
    start_day = start_date.day
    start_year = start_date.year
    

    
    # event_date = datetime.datetime(year, month_converter(month), day, 0, 0, 0)
    start_date = datetime.datetime(start_year,start_month, start_day, start_hour, start_minute, 0) - timedelta(minutes=30)
    
    end_date_string = end_date
    end_month = end_date.month
    end_day = end_date.day
    end_year = end_date.year
    
    # event_date = datetime.datetime(year, month_converter(month), day, 0, 0, 0)
    end_date = datetime.datetime(end_year, end_month, end_day, end_hour, end_minute, 0)
    
    return start_date, end_date



def get_WA_dataset():
    
    start_date = datetime.datetime(2010, month_converter('January'), 1, 0, 0, 0)
    end_date = datetime.datetime(2025, month_converter('December'), 1, 0, 0, 0)
    return start_date, end_date
    
def initialize_event_folders(folder_name, event_id):
    event_folder_id = folder_name + str(event_id) + '/'
    if not os.path.exists(event_folder_id):
        os.makedirs(event_folder_id)
    return event_folder_id



# Domain coordinates for WA 
xmin = -21.4
xmax = 30.4
ymin = -2.9
ymax = 33.1

req_shape = (360, 518)

# Domain coordinates for Ghana 
# xmin = -3.5416666669999999
# xmax = 1.6666666669999999
# ymin = 4.7416666669999987
# ymax = 11.9916666670000005

# xmin = -4.3
# xmax = 2.1
# ymin = 4.7
# ymax = 11.1


# xmin = -3.95
# xmax = 2.35
# ymin = 4.85
# ymax = 11.15

if __name__ == "__main__":
    
    precipBaseFolder = '/vol_efthymios/NFS07/en279/SERVIR/TITO_test3/ML/nowcasting/data/events/'
    # for WA dataset, this returns the start and end dates for the data
    initial_timestamp, final_timestamp = get_WA_dataset()
        
    
    for event_id in range(1, 31):

        initial_timestamp, final_timestamp = get_event(csv_filename='/vol_efthymios/NFS07/en279/SERVIR/TITO_test3/ML/nowcasting/data/flood_events_wa.csv', event_id=event_id)
        
        
        req_height, req_width = req_shape
        
        # final_timestamp = dt.now() - timedelta(minutes=30)
        # initial_timestamp = final_timestamp - timedelta(hours=4)
        
        precipEventFolder = initialize_event_folders(precipBaseFolder, event_id=event_id)
        meta_fname = precipEventFolder + 'metadata.json'
        
        ppt_server_path = 'https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/gis/early/'
        
        email = 'aaravamudan2014@my.fit.edu'
        get_gpm_files(precipEventFolder,initial_timestamp, final_timestamp, ppt_server_path ,email, req_height, req_width)
        
        from servir_data_utils.m_tif2h5py import tif2h5py
        
        tif2h5py(tif_directory=precipEventFolder, h5_fname=precipBaseFolder + str(event_id)+'.h5', meta_fname=meta_fname, x1=xmin, y1=ymin, x2=xmax, y2=ymax)
        