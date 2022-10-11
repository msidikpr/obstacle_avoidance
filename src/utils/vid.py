import numpy as np
from scipy.interpolate import interp1d

def norm_worldcam(arr, time, gamma=2):
    world_norm = (arr/255)**2
    std_im = np.std(world_norm, axis=0)
    std_im[std_im < 10/255] = 10/255
    img_norm = (world_norm - np.mean(world_norm, axis=0)) / std_im
    img_norm = img_norm * (std_im > 20/255)
    img_norm[img_norm < -2] = -2
    movInterp = interp1d(time, img_norm, axis=0, bounds_error=False)