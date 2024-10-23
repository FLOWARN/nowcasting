# to download IMERG early run data from online

# Import all libraries
import sys
import subprocess
import os
import datetime as DT
import osgeo.gdal as gdal
from osgeo.gdal import gdalconst
from osgeo.gdalconst import GA_ReadOnly
#from gdalconst import GA_ReadOnly
#from gdalconst import *
import numpy as np

# Download and process files
server = 'https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/gis/early/'
file_prefix = '3B-HHR-E.MS.MRG.3IMERG.'
file_suffix = '.V06B.30min.tif'
# file_suffix = '.V06C.30min.tif'     # this extension is used for data from May 2023 to present 
# Email associated to PMM account


email = 'vrobledodelgado@uiowa.edu'

def main(argv):
  # Domain coordinates
  xmin = -21.4
  xmax = 30.4
  ymin = -2.9
  ymax = 33.1

# for initial and final date, the date of the water spike is taken,and then a 5 day buffer is given (so date starts 5 days before spike and finishes 5 days after
# And initial date then starts 3 months before that, to give data for EF5 warm up period
  # Initial Date
  year_i = 2023
  month_i = 3
  day_i = 6

  # Final Date
  year_f = 2023
  month_f = 3
  day_f = 9

  # loop through the file list and get each file
  initial_date = DT.datetime(year_i,month_i,day_i,0,0,0)
  final_date = DT.datetime(year_f,month_f,day_f,0,0,0)
  delta_time = DT.timedelta(minutes=30)

  # Loop through dates
  current_date = initial_date
  acumulador_30M = 0

  while (current_date < final_date):
    initial_time_stmp = current_date.strftime('%Y%m%d-S%H%M%S')

    final_time = current_date + DT.timedelta(minutes=29)
    final_time_stmp = final_time.strftime('E%H%M59')
    final_time_gridout = current_date + DT.timedelta(minutes=30)

    folder = current_date.strftime('%Y/%m/')

    date_stamp = initial_time_stmp + '-' + final_time_stmp + '.' + '{:04d}'.format(acumulador_30M)

    filename = folder + file_prefix + date_stamp + file_suffix
    print('Downloading ' + server + '/' + filename)
    try:
        # Download from NASA server
        get_file(filename)

        # Process file for domain and to fit EF5
        # Filename has final datestamp as it represents the accumulation upto that point in time
        gridOutName = 'imerg.' + final_time_gridout.strftime('%Y%m%d%H%M') + '.tif'
        local_filename = file_prefix + date_stamp + file_suffix
        NewGrid, nx, ny, gt, proj = processIMERG(local_filename,xmin,ymin,xmax,ymax)

        # Write out processed filename
        WriteGrid(gridOutName, NewGrid, nx, ny, gt, proj)
    except Exception as e:
        print(e)
        print(filename)

    # Advance in time
    current_date = current_date + delta_time

    # If the day changes, reset acumulador_30M
    if (acumulador_30M < 1410):
      acumulador_30M = acumulador_30M + 30
    else:
      print("New day")
      acumulador_30M = 0

def get_file(filename):
    ''' Get the given file from jsimpsonhttps using curl. '''
    url = server + '/' + filename

    cmd = 'curl -sO -u ' + email + ':' + email + ' ' + url
    args = cmd.split()

    process = subprocess.Popen(args, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    process.wait() # wait so this program doesn't end
                        # before getting all files

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

if __name__ == '__main__':
    main(sys.argv)
