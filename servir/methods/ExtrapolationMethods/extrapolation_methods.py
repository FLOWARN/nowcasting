import time

import numpy as np
from pysteps.utils import transformation
from pysteps import nowcasts
from pysteps import motion
from pysteps.motion.lucaskanade import dense_lucaskanade

def linda(in_precip,timesteps, max_num_features = 15, add_perturbations=False, n_ens_members = 20, return_output=True):

    # Estimate the motion field
    V = dense_lucaskanade(in_precip)
    nowcast_method = nowcasts.get_method("linda")
    
    # The linda nowcast
    forecast = nowcast_method(in_precip, V, timesteps, max_num_features=max_num_features, add_perturbations=add_perturbations, n_ens_members = n_ens_members, return_output=return_output, vel_pert_method=None)
    
    if return_output:
        return forecast
    else:
        return np.nanmean(forecast, axis=0)
    

def steps(in_precip, timesteps, n_ens_members = 20, n_cascade_levels=6, return_output = True):

    R_train, _ = transformation.dB_transform(in_precip, threshold=0.1, zerovalue=-15.0)

    # Set missing values with the fill value
    R_train[~np.isfinite(R_train)] = -15.0

    # Estimate the motion field
    V = dense_lucaskanade(R_train)

    # The STEPS nowcast
    nowcast_method = nowcasts.get_method("steps")
    R_forcast = nowcast_method(R_train, V, timesteps, n_ens_members=n_ens_members, n_cascade_levels=n_cascade_levels,\
                               precip_thr = -10.0, kmperpixel=10, timestep=30, return_output=return_output)

    # Back-transform to rain rates
    R_forecast = transformation.dB_transform(R_forcast, threshold=-10.0, inverse=True)[0]

    if return_output:
        return np.nanmean(R_forcast, axis=0)
    else:
        # the ensemble mean
        R_f_mean = np.nanmean(R_forcast, axis=0)
        
    return R_f_mean


def langragian_persistance(in_precip, timesteps):
    R_train, _ = transformation.dB_transform(in_precip, threshold=0.1, zerovalue=-15.0)

    # Estimate the motion field with Lucas-Kanade
    oflow_method = motion.get_method("LK")
    V = oflow_method(R_train)

    # Extrapolate the last radar observation
    extrapolate = nowcasts.get_method("extrapolation")
    R_train[~np.isfinite(R_train)] =-15.0
    R_f = extrapolate(R_train[-1, :, :], V, timesteps)

    # Back-transform to rain rate
    R_f = transformation.dB_transform(R_f, threshold=-10.0, inverse=True)[0]

    return R_f






