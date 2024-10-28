
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
import time
import pandas as pd

def ReadandWarp(gridFile,xmin,ymin,xmax,ymax):

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
    pixel_size = gt[1]

    #Warp to model resolution and domain extents
    ds = gdal.Warp('', pre_ds, srcNodata=NoData, srcSRS='EPSG:4326', dstSRS='EPSG:4326', dstNodata='-9999', format='VRT', xRes=pixel_size, yRes=-pixel_size, outputBounds=(xmin,ymin,xmax,ymax))

    WarpedGrid = ds.ReadAsArray()
    new_gt = ds.GetGeoTransform()
    new_proj = ds.GetProjection()
    new_nx = ds.GetRasterBand(1).XSize
    new_ny = ds.GetRasterBand(1).YSize

    return WarpedGrid, new_nx, new_ny, new_gt, new_proj

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


def processIMERG(local_filename,llx,lly,urx,ury):
    # Process grid
    # Read and subset grid
    NewGrid, nx, ny, gt, proj = ReadandWarp(local_filename,llx,lly,urx,ury)
    # Scale value
    NewGrid = NewGrid*0.1
    return NewGrid, nx, ny, gt, proj


def get_file(filename,server,email):
   ''' Get the given file from jsimpsonhttps using curl. '''
   url = server + '/' + filename
   cmd = 'curl -sO -u ' + email + ':' + email + ' ' + url
   args = cmd.split()
   process = subprocess.Popen(args, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
   process.wait() # wait so this program doesn't end before getting all files#

def get_gpm_files(precipFolder,initial_timestamp, final_timestamp, ppt_server_path ,email):
    #path server
    server = ppt_server_path
    file_prefix = '3B-HHR-E.MS.MRG.3IMERG.'
    file_suffix = '.V07B.30min.tif'
    
    # Domain coordinates (This part must be changed)
    xmin = -21.4
    xmax = 30.4
    ymin = -2.9
    ymax = 33.1
    
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
            # Download from NASA server
            get_file(filename,server,email)
            # Process file for domain and to fit EF5
            # Filename has final datestamp as it represents the accumulation upto that point in time
            gridOutName = precipFolder+'imerg.qpe.' + final_time_gridout.strftime('%Y%m%d%H%M') + '.30minAccum.tif'
            local_filename = file_prefix + date_stamp + file_suffix
            NewGrid, nx, ny, gt, proj = processIMERG(local_filename,xmin,ymin,xmax,ymax)
            filerm = file_prefix + date_stamp + file_suffix
            # Write out processed filename
            WriteGrid(gridOutName, NewGrid, nx, ny, gt, proj)
            os.remove(filerm)
        except Exception as e:
            print(e)
            print(filename)
            pass

        # Advance in time
        current_date = current_date + delta_time
        

def get_event(csv_filename, event_id=1):
    events_df = pd.read_csv(csv_filename)
    events_df['Event  Recorded'] = pd.to_datetime(events_df['Event  Recorded'], format='mixed')
    event_df = events_df[events_df['Events'] == event_id]
    start_time = event_df['Event  Recorded']
    end_time = (event_df['Event  Recorded'] + timedelta(hours = 24))
    
    
    return datetime.datetime(start_time.iloc[0].year, start_time.iloc[0].month, start_time.iloc[0].day, 0,0,0), datetime.datetime(start_time.iloc[0].year, start_time.iloc[0].month, start_time.iloc[0].day + 1, 0,0,0)

def initialize_event_folders(folder_name, event_id):
    event_folder_id = folder_name + str(event_id) + '/'
    if not os.path.exists(event_folder_id):
        os.makedirs(event_folder_id)
    return event_folder_id

if __name__ == "__main__":
    
    precipBaseFolder = '/vol_efthymios/NFS07/en279/SERVIR/TITO_test3/ML/nowcasting/data/events/'
    event_id=1
    initial_timestamp, final_timestamp = get_event(csv_filename='/vol_efthymios/NFS07/en279/SERVIR/TITO_test3/ML/nowcasting/data/flood_events_ghana.csv', event_id=event_id)
    
    precipEventFolder = initialize_event_folders(precipBaseFolder, event_id=event_id)
    ppt_server_path = 'https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/gis/early/'
    
    email = 'aaravamudan2014@my.fit.edu'
    # get_gpm_files(precipEventFolder,initial_timestamp, final_timestamp, ppt_server_path ,email)
    
    from servir_data_utils.m_tif2h5py import tif2h5py
    tif2h5py(tif_directory=precipEventFolder, h5_fname=precipBaseFolder + str(event_id)+'.h5')