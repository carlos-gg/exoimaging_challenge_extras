# coding: utf-8

import numpy as np
import vip_hci as vip
import os
from vip_hci.var import fit_2dgaussian
from vip_hci.fits import open_fits


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

# NOTE 1: A participant submission must correspond to a single algorithm (the
# goal of the data challenge is to assess the performance of different HCI
# algorithms!). This means that all the datasets for a given sub-challenge
# must be processed homogeneously (ADI or ADI+mSDI)

# NOTE 2: For phase 1, you shall submit N detection maps, corresponding to each 
# dataset in the specific sub-challenge (the true positive rate calculation
# takes into account the total number of injections, so having less
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
        ds_id = str(i + 1) + '.fits'
        try:
            cube = open_fits(os.path.join(data_dir, instru + '_cube_' + ds_id))
            print(instru, "_", i)
        except:
            continue
        pa = open_fits(os.path.join(data_dir,instru+'_pa_' + ds_id))
        psf = open_fits(os.path.join(data_dir,instru+'_psf_' + ds_id))
        plsc = open_fits(os.path.join(data_dir,instru+'_plsc_' + ds_id))
        if cube.ndim == 4:
            wavelength = open_fits(os.path.join(data_dir,instru+'_wl_' + ds_id))

        # Let's assume we are using a single FWHM (fwhm = 4.8)
        if len(psf.shape) == 3:
            fit = fit_2dgaussian(np.mean(psf, axis=0), full_output=True)
            fwhm = np.mean([fit['fwhm_x'][0], fit['fwhm_y'][0]])
        else:
            fit = fit_2dgaussian(psf, full_output=True)
            fwhm = np.mean([fit['fwhm_x'][0], fit['fwhm_y'][0]])

        # A simple baseline algorithm: median frame subtraction:
        if cube.ndim == 3:
            # Building the stack median frame
            stack_median = np.median(cube, axis=0)
            # Subtracting the median frame from each slice of the sequence
            cube_res = cube - stack_median
            # Rotate each residual slice to align the astrophysical signal
            cube_res_der = vip.preproc.cube_derotate(cube_res, pa)
            # Median combining the residuals
            frame = np.median(cube_res_der, axis=0)
        else:
            vip.medsub.median_sub(cube, pa, wavelength)

        # Creation of a detection map (S/N map in this case)
        detmap = vip.metrics.snrmap(frame, fwhm)

        # Alternatively, you can plug in your own algorithm and produce a
        # detection map.

        # Let's write to disk the detection map as a FTIS file
        vip.fits.write_fits(os.path.join(sub_dir, instru+'_detmap_' + ds_id),
                            detmap)

        # Let's save the FHWM value
        vip.fits.write_fits(os.path.join(sub_dir, instru+'_fwhm_' + ds_id),
                            np.array([fwhm]))


# Let's define our detection threshold and save it
detection_threshold = 4
vip.fits.write_fits(os.path.join(sub_dir, './detection_threshold.fits'),
                    np.array([detection_threshold]))

# Up to this point, we have a list of detection map files and FWHM value:
# instrument_detmap_1.fits
# instrument_detmap_2.fits
# instrument_fwhm_1.fits
# instrument_fwhm_2.fits
# And a file with the detection threshold:
# detection_threshold.fits

# Optionally, you could provide the the parameters npix, overlap_threshold, 
# and max_blob_fact, related to the blob counting procedure. Store them as FITS
# files (eg. npix.fits). Please check out
# https://github.com/carlgogo/exoimaging_challenge_extras.git
# and in particular the notebook DC1_starting_kit.ipynb for a detailed
# explanation of this parameters and the effect on the blob counting procedure.
