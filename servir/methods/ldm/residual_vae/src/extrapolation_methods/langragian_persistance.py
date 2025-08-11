import time
import numpy as np
from pysteps.utils import transformation
from pysteps import nowcasts
from pysteps import motion
from pysteps.motion.lucaskanade import dense_lucaskanade

def langragian_persistance(in_precip, timesteps):
    R_train, _ = transformation.dB_transform(in_precip, threshold=0.1, zerovalue=-15.0)

    # Estimate the motion field with LK
    oflow_method = motion.get_method("LK")
    V = oflow_method(R_train)

    # Extrapolate the last radar observation
    extrapolate = nowcasts.get_method("extrapolation")
    R_train[~np.isfinite(R_train)] =-15.0
    R_f = extrapolate(R_train[-1, :, :], V, timesteps)

    # Back-transform to rain rate
    R_f = transformation.dB_transform(R_f, threshold=-10.0, inverse=True)[0]
    
    # Replace NaNs with last valid frame
    nan_mask = np.isnan(R_f)
    for t in range(R_f.shape[0]):
        R_f[t][nan_mask[t]] = R_train[-1, :, :][nan_mask[t]]

    # Ensure the output is finite
    R_f = np.nan_to_num(R_f, nan=0.0, posinf=0.0, neginf=0.0)
    R_f[R_f < 0] = 0.0  # Ensure no negative values

    return R_f
