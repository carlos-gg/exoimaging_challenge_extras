# coding: utf-8

import numpy as np
import vip_hci as vip # version 0.9.8

# The goal of this starting kit is to to illustrate how to create a submission 
# for phase 1.

# For angular differential imaging (ADI), three FITS files are provided: the 
# image sequence (3d ndarray), the corresponding parallactic angles 
# (1d ndarray), the off-axis PSF (2d ndarray) and the pixel scale for VLT/NACO 
# instrument. 

# Let's assume we have two benchmark datasets witht he same FWHM. 
N = 2

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

for i in range(N):
    cube = vip.fits.open_fits('./instrument_cube_' + str(i + 1) + '.fits') 
    pa = vip.fits.open_fits('./instrument_angles_' + str(i + 1) + '.fits')
    psf = vip.fits.open_fits('./instrument_psf_' + str(i + 1) + '.fits')
    plsc = vip.fits.open_fits('./instrument_plsc_' + str(i + 1) + '.fits')
    fwhm = 4.8

    # A simple baseline algorithm. The subtraction of the median frame:
    # Building the stack median frame
    stack_median = np.median(cube, axis=0)
    # Subtracting the median frame from each slice of the sequence
    cube_res = cube - stack_median
    # Rotate each residual slice to align the astrophysical signal
    cube_res_der = vip.preproc.cube_derotate(cube_res, pa)
    # Median combining the residuals
    frame = np.median(cube_res_der, axis=0)

    detmap = vip.metrics.snrmap(frame, fwhm)

    # Alternatively, you can plug in your algorithm, instead of the baseline + 
    # S/N map calculation, to produce a detection map.

    # Let's write to disk the detection map as a FTIS file
    vip.fits.write_fits('./instrument_detmap_' + str(i + 1) + '.fits', detmap)


# Let's define our detection threshold and save it
detection_threshold = 6
vip.fits.write_fits('./detection_threshold.fits', detection_threshold)
# Let's save the FHWM value
vip.fits.write_fits('./fwhm.fits', fwhm)

# Up to this point, we have a list of detection map files:
# instrument_detmap_1.fits
# instrument_detmap_2.fits
# And two files with the detection threshold and the FWHM used:
# detection_threshold.fits
# fwhm.fits
# Optionally, you could provide the the parameters npix, overlap_threshold, 
# and max_blob_fact, related to the blob counting procedure. Store them as FITS
# files (eg. npix.fits). Please check out the notebook DC1_starting_kit.ipynb 
# for a detailed explanation of this parameters and the effect on the blob 
# counting procedure. 
