import os
import glob
import shutil
import datetime

import imageio
from pysteps.visualization.animations import animate

def create_precipitation_plots(precipitations,timestamps_obs, timestep_min, geodata, path_outputs, title=''):
    """create gif file of precipitation. 
    This function contains two steps:
    1. create png files of precipitation using pysteps.visualization.animations.animate.
    2. load png files and create gif file using imageio.mimsave

    Args:
        precipitations (ndarray): sequence of precipitation in shape of [T, H, W]
        timestamps_obs (list of datetimes): List of datetime objects corresponding to the timestamps of the fields in precipitations.
        timestep_min (float): The time resolution in minutes of the forecast.
        geodata (dictionary): Dictionary containing geographical information about the field.
        path_outputs (str): path to save the gif file
        title (str): title of the gif file
        gif_dur (int, optional): The duration (in seconds) of each frame. Defaults to 1000.
    """

    animate(precipitations, timestamps_obs  = timestamps_obs,
            timestep_min = timestep_min, geodata=geodata, title=title, \
            savefig=True, fig_dpi=300, fig_format='png', path_outputs=path_outputs)
    



def create_precipitation_gif(precipitations,timestamps_obs, timestep_min, geodata, path_outputs, title='', gif_dur = 1000):
    """create gif file of precipitation. 
    This function contains two steps:
    1. create png files of precipitation using pysteps.visualization.animations.animate.
    2. load png files and create gif file using imageio.mimsave

    Args:
        precipitations (ndarray): sequence of precipitation in shape of [T, H, W]
        timestamps_obs (list of datetimes): List of datetime objects corresponding to the timestamps of the fields in precipitations.
        timestep_min (float): The time resolution in minutes of the forecast.
        geodata (dictionary): Dictionary containing geographical information about the field.
        path_outputs (str): path to save the gif file
        title (str): title of the gif file
        gif_dur (int, optional): The duration (in seconds) of each frame. Defaults to 1000.
    """
    temp_path = os.path.join(path_outputs, 'temp')
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    animate(precipitations, timestamps_obs = timestamps_obs,
            timestep_min = timestep_min, geodata=geodata, title=title, \
            savefig=True, fig_dpi=300, fig_format='png', path_outputs=temp_path)
    

    # load images to create .gif file
    images = []
    forcast_precip_imgs = sorted(glob.glob(f"{temp_path}/*.png") )
    for img in forcast_precip_imgs:
        images.append(imageio.imread(img))

    kargs = { 'duration': gif_dur }
    imageio.mimsave(f"{path_outputs}/{title}.gif", images, **kargs)

    # remove temp folder
    shutil.rmtree(temp_path)


