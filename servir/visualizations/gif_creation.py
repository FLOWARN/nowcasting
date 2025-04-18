# -*- coding: utf-8 -*-
import os
import glob
import shutil
import matplotlib
import matplotlib.pyplot as plt
from pysteps.visualization.precipfields import plot_precip_field
from PIL import Image
import imageio
import matplotlib.gridspec as gridspec


"""
pysteps.visualization.animations copy for local edits
================================

Functions to produce animations for pysteps.

.. autosummary::
    :toctree: ../generated/

    animate
"""

import os
import warnings

import matplotlib.pylab as plt
import pysteps as st

PRECIP_VALID_TYPES = ("ensemble", "mean", "prob")
MOTION_VALID_METHODS = ("quiver", "streamplot")


def animate(
    precip_obs,
    precip_fct=None,
    timestamps_obs=None,
    timestep_min=None,
    motion_field=None,
    ptype="ensemble",
    motion_plot="quiver",
    geodata=None,
    title=None,
    prob_thr=None,
    display_animation=True,
    nloops=1,
    time_wait=0.2,
    savefig=False,
    fig_dpi=100,
    fig_format="png",
    path_outputs="",
    precip_kwargs=None,
    motion_kwargs=None,
    map_kwargs=None,
):
    """
    Function to animate observations and forecasts in pysteps.

    It also allows to export the individual frames as figures, which
    is useful for constructing animated GIFs or similar.

    .. _Axes: https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes

    Parameters
    ----------
    precip_obs: array-like
        Three-dimensional array containing the time series of observed
        precipitation fields.
    precip_fct: array-like, optional
        The three or four-dimensional (for ensembles) array
        containing the time series of forecasted precipitation field.
    timestamps_obs: list of datetimes, optional
        List of datetime objects corresponding to the time stamps of
        the fields in precip_obs.
    timestep_min: float, optional
        The time resolution in minutes of the forecast.
    motion_field: array-like, optional
        Three-dimensional array containing the u and v components of
        the motion field.
    motion_plot: string, optional
        The method to plot the motion field. See plot methods in
        :py:mod:`pysteps.visualization.motionfields`.
    geodata: dictionary or None, optional
        Dictionary containing geographical information about
        the field.
        If geodata is not None, it must contain the following key-value pairs:

        .. tabularcolumns:: |p{1.5cm}|L|

        +----------------+----------------------------------------------------+
        |        Key     |                  Value                             |
        +================+====================================================+
        |   projection   | PROJ.4-compatible projection definition            |
        +----------------+----------------------------------------------------+
        |    x1          | x-coordinate of the lower-left corner of the data  |
        |                | raster                                             |
        +----------------+----------------------------------------------------+
        |    y1          | y-coordinate of the lower-left corner of the data  |
        |                | raster                                             |
        +----------------+----------------------------------------------------+
        |    x2          | x-coordinate of the upper-right corner of the data |
        |                | raster                                             |
        +----------------+----------------------------------------------------+
        |    y2          | y-coordinate of the upper-right corner of the data |
        |                | raster                                             |
        +----------------+----------------------------------------------------+
        |    yorigin     | a string specifying the location of the first      |
        |                | element in the data raster w.r.t. y-axis:          |
        |                | 'upper' = upper border, 'lower' = lower border     |
        +----------------+----------------------------------------------------+

    title: str or None, optional
        If not None, print the string as title on top of the plot.
    ptype: {'ensemble', 'mean', 'prob'}, str, optional
        Type of the plot to animate. 'ensemble' = ensemble members,
        'mean' = ensemble mean, 'prob' = exceedance probability
        (using threshold defined in prob_thrs).
    prob_thr: float, optional
        Intensity threshold for the exceedance probability maps. Applicable
        if ptype = 'prob'.
    display_animation: bool, optional
        If set to True, display the animation (set to False if only
        interested in saving the animation frames).
    nloops: int, optional
        The number of loops in the animation.
    time_wait: float, optional
        The time in seconds between one frame and the next. Applicable
        if display_animation is True.
    savefig: bool, optional
        If set to True, save the individual frames into path_outputs.
    fig_dpi: float, optional
        The resolution in dots per inch. Applicable if savefig is True.
    fig_format: str, optional
        Filename extension. Applicable if savefig is True.
    path_outputs: string, optional
        Path to folder where to save the frames. Applicable if savefig is True.
    precip_kwargs: dict, optional
        Optional parameters that are supplied to
        :py:func:`pysteps.visualization.precipfields.plot_precip_field`.
    motion_kwargs: dict, optional
        Optional parameters that are supplied to
        :py:func:`pysteps.visualization.motionfields.quiver` or
        :py:func:`pysteps.visualization.motionfields.streamplot`.
    map_kwargs: dict, optional
        Optional parameters that need to be passed to
        :py:func:`pysteps.visualization.basemaps.plot_geography`.

    Returns
    -------
    None
    """

    if precip_kwargs is None:
        precip_kwargs = {}

    if motion_kwargs is None:
        motion_kwargs = {}

    if map_kwargs is None:
        map_kwargs = {}

    if precip_fct is not None:
        if len(precip_fct.shape) == 3:
            precip_fct = precip_fct[None, ...]
        n_lead_times = precip_fct.shape[1]
        n_members = precip_fct.shape[0]
    else:
        n_lead_times = 0
        n_members = 1

    if title is not None and isinstance(title, str):
        title_first_line = title + "\n"
    else:
        title_first_line = ""

    if motion_plot not in MOTION_VALID_METHODS:
        raise ValueError(
            f"Invalid motion plot method '{motion_plot}'."
            f"Supported: {str(MOTION_VALID_METHODS)}"
        )

    if ptype not in PRECIP_VALID_TYPES:
        raise ValueError(
            f"Invalid precipitation type '{ptype}'."
            f"Supported: {str(PRECIP_VALID_TYPES)}"
        )

    if timestamps_obs is not None:
        if len(timestamps_obs) != precip_obs.shape[0]:
            raise ValueError(
                f"The number of timestamps does not match the size of precip_obs: "
                f"{len(timestamps_obs)} != {precip_obs.shape[0]}"
            )
        if precip_fct is not None:
            reftime_str = timestamps_obs[-1].strftime("%Y%m%d%H%M")
        else:
            reftime_str = timestamps_obs[0].strftime("%Y%m%d%H%M")
    else:
        reftime_str = None

    if ptype == "prob" and prob_thr is None:
        raise ValueError("ptype 'prob' needs a prob_thr value")

    if ptype != "ensemble":
        n_members = 1

    n_obs = precip_obs.shape[0]

    loop = 0
    while loop < nloops:
        for n in range(n_members):
            for i in range(n_obs + n_lead_times):
                plt.clf()

                # Observations
                if i < n_obs and (display_animation or n == 0):
                    title = title_first_line 
                    if timestamps_obs is not None:
                        title += (
                            f"  {timestamps_obs[i].strftime('%Y-%m-%d %H:%M')}"
                        )

                    plt.clf()
                    if ptype == "prob":
                        prob_field = st.postprocessing.ensemblestats.excprob(
                            precip_obs[None, i, ...], prob_thr
                        )
                        plt.axis('off')
                        ax = st.plt.plot_precip_field(
                            prob_field,
                            ptype="prob",
                            geodata=geodata,
                            probthr=prob_thr,
                            title=title,
                            map_kwargs=map_kwargs,
                            axis='off',
                            **precip_kwargs,
                        )
                    else:
                        plt.axis('off')
                        ax = st.plt.plot_precip_field(
                            precip_obs[i, :, :],
                            geodata=geodata,
                            title=title,
                            map_kwargs=map_kwargs,
                            axis='off',
                            **precip_kwargs,
                        )

                    if motion_field is not None:
                        if motion_plot == "quiver":
                            st.plt.quiver(
                                motion_field, ax=ax, geodata=geodata, **motion_kwargs
                            )
                        elif motion_plot == "streamplot":
                            st.plt.streamplot(
                                motion_field, ax=ax, geodata=geodata, **motion_kwargs
                            )

                    if savefig & (loop == 0):
                        figtags = [reftime_str, ptype, f"f{i:02d}"]
                        figname = "_".join([tag for tag in figtags if tag])
                        filename = os.path.join(path_outputs, f"{figname}.{fig_format}")
                        plt.savefig(filename, bbox_inches="tight", dpi=fig_dpi)
                        print("saved: ", filename)

                # Forecasts
                elif i >= n_obs and precip_fct is not None:
                    title = title_first_line + "Forecast"
                    if timestamps_obs is not None:
                        title += f" {timestamps_obs[-1].strftime('%Y-%m-%d %H:%M')}"
                    if timestep_min is not None:
                        title += " +%02d min" % ((1 + i - n_obs) * timestep_min)
                    else:
                        title += " +%02d" % (1 + i - n_obs)

                    plt.clf()
                    if ptype == "prob":
                        prob_field = st.postprocessing.ensemblestats.excprob(
                            precip_fct[:, i - n_obs, :, :], prob_thr
                        )
                        plt.axis('off')
                        ax = st.plt.plot_precip_field(
                            prob_field,
                            ptype="prob",
                            geodata=geodata,
                            probthr=prob_thr,
                            title=title,
                            map_kwargs=map_kwargs,
                            axis='off',
                            **precip_kwargs,
                        )
                    elif ptype == "mean":
                        ens_mean = st.postprocessing.ensemblestats.mean(
                            precip_fct[:, i - n_obs, :, :]
                        )
                        plt.axis('off')
                        ax = st.plt.plot_precip_field(
                            ens_mean,
                            geodata=geodata,
                            title=title,
                            map_kwargs=map_kwargs,
                            **precip_kwargs,
                            axis='off'
                        )
                    else:
                        plt.axis('off')
                        ax = st.plt.plot_precip_field(
                            precip_fct[n, i - n_obs, ...],
                            geodata=geodata,
                            title=title,
                            map_kwargs=map_kwargs,
                            axis='off',
                            **precip_kwargs,
                        )

                    if motion_field is not None:
                        if motion_plot == "quiver":
                            st.plt.quiver(
                                motion_field, ax=ax, geodata=geodata, **motion_kwargs
                            )
                        elif motion_plot == "streamplot":
                            st.plt.streamplot(
                                motion_field, ax=ax, geodata=geodata, **motion_kwargs
                            )

                    if ptype == "ensemble" and n_members > 1:
                        plt.text(
                            0.01,
                            0.99,
                            "m %02d" % (n + 1),
                            transform=ax.transAxes,
                            ha="left",
                            va="top",
                        )

                    if savefig & (loop == 0):
                        figtags = [reftime_str, ptype, f"f{i:02d}", f"m{n + 1:02d}"]
                        figname = "_".join([tag for tag in figtags if tag])
                        filename = os.path.join(path_outputs, f"{figname}.{fig_format}")
                        plt.axis('off')
                        plt.savefig(filename, bbox_inches="tight", dpi=fig_dpi)
                        print("saved: ", filename)

                if display_animation:
                    plt.pause(time_wait)

            if display_animation:
                plt.pause(2 * time_wait)

        loop += 1

    plt.close()


def create_precipitation_plots(precipitations, labels, timestamps_obs, timestep_min, geodata, path_outputs, title=''):
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

    animate(precipitations, timestamps_obs = timestamps_obs, ptype='ensemble',
            timestep_min = timestep_min, geodata=geodata, title=title, \
            savefig=True, fig_dpi=50, fig_format='png', path_outputs=temp_path)
    

    # load images to create .gif file
    images = []
    forcast_precip_imgs = sorted(glob.glob(f"{temp_path}/*.png") )
    for img in forcast_precip_imgs:
        loaded_image = imageio.imread(img)
        images.append(loaded_image)

    kargs = { 'duration': gif_dur }
    imageio.mimsave(f"{path_outputs}/{title}.gif", images, **kargs)

    # remove temp folder
    shutil.rmtree(temp_path)


def create_precipitation_gif_multiple_comparison(precipitations,labels, timestamps_obs, timestep_min, geodata, path_outputs, title='', gif_dur = 1000):
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
    temp_path = path_outputs + 'temp/'
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    
    # go through number of predictions
    for j in range(len(precipitations[0])):    
        fig = plt.figure()
        # gs = gridspec.GridSpec(1, len(precipitations))
        # go through each of the prediction type (convlstm, lagrangian ,...)
        for i in range(len(precipitations)):
            plt.subplot(1,len(precipitations), i + 1)
            plt.axis('off')
            
            if i ==len(precipitations) - 1:
                colorbar = True
            else:
                colorbar = False
            ax = plot_precip_field(precipitations[i][j], geodata=geodata, colorbar=colorbar, axis='off')
            ax.set_title(labels[i] , fontsize=5)
        fig.suptitle(timestamps_obs[j].decode("utf-8"))
        # gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])  
        plt.subplots_adjust(top=1.5)
        plt.savefig(temp_path + str(j)+'.png', bbox_inches="tight", dpi=200)        

        
    # load images to create .gif file
    images = []
    forcast_precip_imgs = sorted(glob.glob(f"{temp_path}/*.png") )
    for img in forcast_precip_imgs:
        loaded_image = imageio.imread(img)
        # if loaded_image.shape[0] == 1287:
        images.append(loaded_image)

    kargs = { 'duration': gif_dur }
    imageio.mimsave(f"{path_outputs}/{title}.gif", images, **kargs)

    # remove temp folder
    shutil.rmtree(temp_path)
    

def create_static_plot(precipitations, model_names):
    img = Image.open('temp.png')

    plt.clf()
    global_min_non_zero_precipitation_intensity = 0.0 # mm/h
    global_max_precipitation_intensity = 20.0 # mm/h
    geodata_dict['yorigin'] = 'upper'


        
    img_size = (64,64)
    # Assumes the input dimensions are lat/lon
    # nlat, nlon = img_size

    # x_grid, y_grid, extent, regular_grid, origin = get_geogrid(
    #     nlat, nlon, geodata=geodata_dict
    # )

    cmap_custom = matplotlib.cm.Blues
    captions = [ -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, +0.5, +1, +1.5, +2]
    # Use white color for pixels whose intensity is below 
    # global_min_non_zero_precipitation_intensity.
    # Use pink color for pixels whose intensity is above 
    # global_max_precipitation_intensity.
    cmap_custom.set_under('white') 
    cmap_custom.set_over('pink')


    # fig, axarr = plt.subplots(12,2, figsize = (36, 256), gridspec_kw = {'wspace':0.07, 'hspace':-0.5})

    fig = plt.figure(figsize = (12, 60))

    image_index = 666


    # data[data < 0] = 0
    # gt_data = y[:,:,0,:,:][0]
    # input_data = x[:,:,0,:,:][0]


    for i in range(0,12):
        # axarr[i+1][0].imshow(img, alpha=0.5)
        # im = axarr[i+1][0].imshow(img, alpha=0.25,aspect='auto' )
        # ax = fig.add_subplot(821 + ((i+1)*2))
        # im = axarr[i+1][0].imshow(gt_data[i], animated=False, cmap=cmap_custom,alpha=0.95)
        plt.subplot(12,3,3*i+1)
        plt.axis('off')
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # fig.patch.set_visible(True)
        # ax.get_xaxis().set_ticks([])
        # ax.get_yaxis().set_ticks([])
        ax = plot_precip_field(gt_data[image_index][i][0], geodata=geodata_dict, colorbar=False, axis='off')
        
        if i == 0:
            title_string_1 = str(captions[i]) + " H \n Satellite Estimate"
            title_string_2 = str(captions[i]) + "H \n DGMR Prediction"
            title_string_3 = str(captions[i]) + "H \n DGMR + IR Prediction"
        else:
            title_string_1 = str(captions[i]) + " H"
            title_string_2 = str(captions[i]) + " H"
            title_string_3 = str(captions[i]) + " H"
        
        ax.set_title(title_string_1 , fontsize=20)
        # ax = fig.add_subplot(822 + ((i+1)*2))
        # ax.axis('off')
        # # ax.get_xaxis().set_ticks([])
        # # ax.get_yaxis().set_ticks([])
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # fig.patch.set_visible(True)
        plt.subplot(12,3,3*(i) + 2)
        plt.axis('off')
        ax = plot_precip_field(data[image_index][i][0], geodata=geodata_dict, colorbar=False, axis='off')
        
        ax.set_title(title_string_2 , fontsize=20)
        
        plt.subplot(12,3,3*(i) + 3)
        plt.axis('off')
        ax = plot_precip_field(data_ir[image_index][i][0], geodata=geodata_dict, colorbar=False, axis='off')
        ax.set_title(title_string_3 , fontsize=20)
        
        # im = axarr[i+1][2].imshow(data_ir[image_index][i][0], animated=False, cmap=cmap_custom)
        # axarr[i+1][2].set_title(title_string_3 , fontsize=70)
        
    # fig.suptitle('Time steps $T_{'+str(captions[0])+' H}$ to $T_{'+str(captions[-1])+' H}$', fontsize=50,y=0.38)

    # axarr[0][0].imshow(input_data[image_index][-1][0], animated=False, cmap=cmap_custom)
    # axarr[0][1].imshow(input_data[image_index][-1][0], animated=False, cmap=cmap_custom)
    # axarr[0][2].imshow(input_data[image_index][-1][0], animated=False, cmap=cmap_custom)

    cax = fig.add_axes([0.023, 0.45, 0.01, 0.15])
    cbar = fig.colorbar(ax.get_images()[0],cax=cax, orientation='vertical')
    cbar.ax.tick_params(labelsize=25)

    plt.savefig('results_highres.png', bbox_inches="tight", dpi=300)

    plt.show()
    plt.close()
    