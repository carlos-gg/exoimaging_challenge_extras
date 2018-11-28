# coding: utf-8

import numpy as np
import vip_hci as vip # version 0.9.8
import os
from vip_hci.var import fit_2dgaussian


# The goal of this starting kit is to to illustrate how to create a submission 
# for phase 1.

# For angular differential imaging (ADI), three FITS files are provided: the 
# image sequence (3d ndarray), the corresponding parallactic angles 
# (1d ndarray), the off-axis PSF (2d ndarray) and the pixel scale for VLT/NACO 
# instrument. 

# Let's assume we have three benchmark datasets witht the same FWHM. 
N = 3

# We load the datasets into Numpy ndarrays and process it with a baseline 
# algorithm to generate a detection map.

# NOTE 1: A participant must correspond to a single algorithm (the goal of the 
# data challenge is to assess the performance of algorithms!). Of course you
# can create several participants with different algorithmic approaches.

# NOTE 2: For phase 1, you shall submit N detection maps, corresponding to each 
# dataset in the challenge competition library (the true positive rate 
# calculation takes into account the total number of injections, so having less 
# detection maps will penalize your result). Also, you must submit a single 
# detection threshold which depends on the statistics of the algorithm/detection 
# map, and finally the value of the full width at half maximum used.

# We assume a naming convention: instrument_{cube/pa/psf}_id.fits
list_instru = ["sphere_irdis","nirc2"]
data_dir = "../../data/public_data/"
sub_dir = "../submission"
if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
for instru in list_instru:
    for i in range(N):
        try:
                cube = vip.fits.open_fits(os.path.join(data_dir,instru+'_cube_' + str(i + 1) + '.fits'))
                print(instru,"_",i)
        except:
                continue
        pa = vip.fits.open_fits(os.path.join(data_dir,instru+'_pa_' + str(i + 1) + '.fits'))
        psf = vip.fits.open_fits(os.path.join(data_dir,instru+'_psf_' + str(i + 1) + '.fits'))
        plsc = vip.fits.open_fits(os.path.join(data_dir,instru+'_plsc_' + str(i + 1) + '.fits'))
        if cube.ndim==4:
                wavelength= vip.fits.open_fits(os.path.join(data_dir,instru+'_wl_' + str(i + 1) + '.fits'))

        # Let's assume we are using a single FWHM
        #fwhm = 4.8
        if len(psf.shape) ==3:
                fwhm = np.mean([fit_2dgaussian(np.mean(psf, axis=0), full_output=True)['fwhm_x'][0],
                        fit_2dgaussian(np.mean(psf, axis=0), full_output=True)['fwhm_y'][0]])
        else:
                fwhm = np.mean([fit_2dgaussian(psf, full_output=True)['fwhm_x'][0],
                        fit_2dgaussian(psf, full_output=True)['fwhm_y'][0]])

        if cube.ndim==3:

                # A simple baseline algorithm. The subtraction of the median frame:
                # Building the stack median frame
                stack_median = np.median(cube, axis=0)
                # Subtracting the median frame from each slice of the sequence
                cube_res = cube - stack_median
                # Rotate each residual slice to align the astrophysical signal
                cube_res_der = vip.preproc.cube_derotate(cube_res, pa)
                # Median combining the residuals
                frame = np.median(cube_res_der, axis=0)
        else:
                vip.medsub.median_sub(cube,pa,wavelength)
        detmap = vip.metrics.snrmap(frame, fwhm)

        # Alternatively, you can plug in your algorithm, instead of the baseline + 
        # S/N map calculation, to produce a detection map.

        # Let's write to disk the detection map as a FTIS file
        vip.fits.write_fits(os.path.join(sub_dir,instru+'_detmap_' + str(i + 1) + '.fits'), detmap)

        # Let's save the FHWM value
        vip.fits.write_fits(os.path.join(sub_dir,instru+'_fwhm_' + str(i + 1) + '.fits'), np.array([fwhm]))


# Let's define our detection threshold and save it
detection_threshold = 4
vip.fits.write_fits(os.path.join(sub_dir,'./detection_threshold.fits'), np.array([detection_threshold]))

# Up to this point, we have a list of detection map files and FWHM value:
# instrument_detmap_1.fits
# instrument_detmap_2.fits
# instrument_fwhm_1.fits
# instrument_fwhm_2.fits
# And two files with the detection threshold:
# detection_threshold.fits

# Optionally, you could provide the the parameters npix, overlap_threshold, 
# and max_blob_fact, related to the blob counting procedure. Store them as FITS
# files (eg. npix.fits). Please check out https://github.com/carlgogo/exoimaging_challenge_extras.git 
# and in particular the notebook DC1_starting_kit.ipynb for a detailed explanation of this parameters 
# and the effect on the blob  counting procedure. 
