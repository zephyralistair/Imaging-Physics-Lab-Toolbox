import numpy as np
from scipy.ndimage import uniform_filter

def decay_correction(PET_arrays, half_life, Tis):
    """Decay correction for dynamic PET.

    Args:
        PET_arrays: dynamic PET, shape (n_frames, x, y, z)
        half_life: half-life of the radiotracer (in minutes)
        Tis: time of each frame (in minutes)

    Returns:
        Decay-corrected dynamic PET

    """
    # decay correct the PET image
    decay_lambda = np.log(2) / half_life
    decay_correction_factor = np.exp(-decay_lambda * Tis)
    return PET_arrays * decay_correction_factor[:, np.newaxis, np.newaxis, np.newaxis]

def decay_uncorrection(PET_arrays, half_life, Tis):
    """Decay uncorrection for dynamic PET.

    Args:
        PET_arrays: dynamic PET, shape (n_frames, x, y, z)
        half_life: half-life of the radiotracer (in minutes)
        Tis: time of each frame (in minutes)

    Returns:
        Decay-uncorrected dynamic PET

    """
    # decay uncorrect the PET image
    decay_lambda = np.log(2) / half_life
    decay_correction_factor = np.exp(-decay_lambda * Tis)
    return PET_arrays / decay_correction_factor[:, np.newaxis, np.newaxis, np.newaxis]

def hypr_lr(PET_arrays, Tis, dts, smooth_voxel_nums, half_life):
    """HYPR-LR denoising for dynamic PET.

    HYPR-LR is short for HighlY constrained backPRojection for Local Reconstruction.

    Reference:
    Christian, B. T., Vandehey, N. T., Floberg, J. M., Mistretta, C. A. (2010).
    Dynamic PET denoising with HYPR processing. Journal of Nuclear Medicine, 51(7),
    1147-1154. https://doi.org/10.2967/jnumed.109.073999

    Args:
        PET_arrays: dynamic PET, shape (n_frames, x, y, z)
        Tis: time of each frame (in minutes)
        dts: duration of each frame (in minutes)
        smooth_voxel_nums: number of voxels to smooth over (size of kernel)
        half_life: half-life of the radiotracer (in minutes)

    Returns:
        HYPR-LR denoised dynamic PET

    """
    # decay uncorrect the PET image
    i = decay_uncorrection(PET_arrays, half_life, Tis)

    # calculate duration-weighted frame average
    ic = np.average(i, axis=0, weights=dts)

    # convolve both ti and weighted average by a low-pass filter (3D boxcar)
    i_smoothed = uniform_filter(i, size = smooth_voxel_nums, axes = (1, 2, 3))
    ic_smoothed = uniform_filter(ic,  size = smooth_voxel_nums, axes = (0, 1, 2))

    # add small number to the denominator to prevent zero division
    eps = 1e-8
    iw = i_smoothed / (ic_smoothed[np.newaxis, ...] + eps)
    ih = iw * ic[np.newaxis, ...]

    # decay correct the result
    return decay_correction(ih, half_life, Tis)