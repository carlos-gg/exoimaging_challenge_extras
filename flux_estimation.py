"""
Generation of labeled data for supervised learning. To be used to train the
discriminative models. 
"""
from __future__ import print_function, division, absolute_import

__all__ = ['estimate_fluxes']

import os
import tables
import copy
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from vip_hci.conf import time_ini, timing, time_fin
from vip_hci.var import frame_center
from vip_hci.stats import frame_average_radprofile
from vip_hci.conf.utils_conf import (pool_map, fixed, make_chunks)
from vip_hci.var import cube_filter_highpass, pp_subplots, get_annulus_segments
from vip_hci.metrics import cube_inject_companions
from vip_hci.preproc import (check_pa_vector, cube_derotate, cube_crop_frames,
                             frame_rotate, frame_shift, frame_px_resampling,
                             frame_crop)
from vip_hci.preproc.derotation import _compute_pa_thresh, _find_indices_adi
from vip_hci.metrics import frame_quick_report
from vip_hci.medsub import median_sub
from vip_hci.pca import pca


def estimate_fluxes(cube, psf, distances, angles, fwhm, plsc, wavelengths=None,
                    n_injections=10, min_adi_snr=2, max_adi_snr=5,
                    random_seed=42, kernel='rbf', epsilon=0.1, c=1e4,
                    gamma=1e-2, figsize=(10, 5), dpi=100, n_proc=2, **kwargs):
    """
    Automatic estimation of the scaling factors for injecting the
    companions (brightness or contrast of fake companions).

    Epsilon-Support Vector Regression (important parameters in the model are
    C and epsilon). The implementation is based on scikit-learn (libsvm).

    Parameters
    ----------
    C : float, optional (default=1.0)
        Penalty parameter C of the error term.
    epsilon : float, optional (default=0.1)
        Epsilon in the epsilon-SVR model. It specifies the epsilon-tube
        within which no penalty is associated in the training loss function
        with points predicted within a distance epsilon from the actual
        value.
    kernel : string, optional (default=’rbf’)
        Specifies the kernel type to be used in the algorithm. It must be
        one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a
        callable. If none is given, ‘rbf’ will be used. If a callable is
        given it is used to precompute the kernel matrix.
    gamma : float, optional (default=’auto’)
        Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. If gamma is
        ‘auto’ then 1/n_features will be used instead.

    """
    starttime = time_ini()

    global GARRAY
    GARRAY = cube
    global GARRPSF
    GARRPSF = psf
    global GARRPA
    GARRPA = angles
    global GARRWL
    GARRWL = wavelengths

    if cube.ndim == 4:
        if wavelengths is None:
            raise ValueError('`wavelengths` parameter must be provided')

    # Getting the radial profile in the mean frame of the cube
    sampling_sep = 1
    radius_int = 1
    if cube.ndim == 3:
        global_frame = np.mean(cube, axis=0)
    elif cube.ndim == 4:
        global_frame = np.mean(cube.reshape(-1, cube.shape[2], cube.shape[3]),
                               axis=0)

    me = frame_average_radprofile(global_frame, sep=sampling_sep,
                                  init_rad=radius_int, plot=False)
    radprof = np.array(me.radprof)
    radprof = radprof[np.array(distances) + 1]

    flux_min = radprof * 0.1
    flux_min[flux_min < 0] = 0.1

    # Multiprocessing pool
    flux_max = pool_map(n_proc, _get_max_flux, fixed(range(len(distances))),
                        distances, radprof, fwhm, plsc, max_adi_snr,
                        wavelengths)
    flux_max = np.array(flux_max)
    fluxes_list, snrs_list = _sample_flux_snr(distances, fwhm, plsc,
                                              n_injections, flux_min,
                                              flux_max, n_proc, random_seed,
                                              wavelengths)

    plotvlines = [min_adi_snr, max_adi_snr]
    nsubplots = len(distances)
    if nsubplots % 2 != 0:
        nsubplots -= 1
    ncols = 4
    nrows = int(nsubplots / ncols) + 1

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi,
                            sharey='row')
    fig.subplots_adjust(wspace=0.05, hspace=0.3)
    axs = axs.ravel()
    fhi = list()
    flo = list()

    # Regression for each distance
    for i, d in enumerate(distances):
        fluxes = np.array(fluxes_list[i])
        snrs = np.array(snrs_list[i])
        mask = np.where(snrs > 0.1)
        snrs = snrs[mask].reshape(-1, 1)
        fluxes = fluxes[mask].reshape(-1, 1)

        model = SVR(kernel=kernel, epsilon=epsilon, C=c, gamma=gamma,
                    **kwargs)
        model.fit(X=snrs, y=fluxes)
        flux_for_lowsnr = model.predict(min_adi_snr)
        flux_for_higsnr = model.predict(max_adi_snr)
        fhi.append(flux_for_higsnr[0])
        flo.append(flux_for_lowsnr[0])
        snrminp = min_adi_snr / 2
        snrs_pred = np.linspace(snrminp, max_adi_snr + snrminp,
                                num=50).reshape(-1, 1)
        fluxes_pred = model.predict(snrs_pred)

        # Figure of flux vs s/n
        axs[i].xaxis.set_tick_params(labelsize=6)
        axs[i].yaxis.set_tick_params(labelsize=6)
        axs[i].plot(fluxes, snrs, '.', alpha=0.2, markersize=4)
        axs[i].plot(fluxes_pred, snrs_pred, '-', alpha=0.99,
                    label='S/N regression model', color='orangered')
        axs[i].grid(which='major', alpha=0.3)
        axs[i].legend(fontsize=6)
        for l in plotvlines:
            axs[i].plot((0, max(fluxes)), (l, l), ':', color='darksalmon')
        ax0 = fig.add_subplot(111, frame_on=False)
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.set_xlabel('Fakecomp flux scaling [Counts]', labelpad=25,
                       size=8)
        ax0.set_ylabel('ADI-medsub median S/N (3 equidist. angles)',
                       labelpad=25, size=8)

    for i in range(len(distances), len(axs)):
        axs[i].axis('off')

    timing(starttime)

    flo = np.array(flo).flatten()
    fhi = np.array(fhi).flatten()

    # x = distances
    # f1 = interpolate.interp1d(x, flo, fill_value='extrapolate')
    # f2 = interpolate.interp1d(x, fhi, fill_value='extrapolate')
    # fhi = f2(distances_init)
    # flo = f1(distances_init)

    plt.figure(figsize=(10, 4), dpi=dpi)
    plt.plot(distances, radprof, '--', alpha=0.8, color='gray', lw=2,
             label='average radial profile')
    plt.plot(distances, flo, '.-', alpha=0.6, lw=2, color='dodgerblue',
             label='flux lower interval')
    plt.plot(distances, fhi, '.-', alpha=0.6, color='dodgerblue', lw=2,
             label='flux upper interval')
    plt.fill_between(distances, flo, fhi, where=flo <= fhi, alpha=0.2,
                     facecolor='dodgerblue', interpolate=True)
    plt.grid(which='major', alpha=0.4)
    plt.xlabel('Distance from the center [Pixels]')
    plt.ylabel('Fakecomp flux scaling [Counts]')
    plt.minorticks_on()
    plt.xlim(0)
    plt.ylim(0)
    plt.legend()
    plt.show()
    return flo, fhi


def _get_max_flux(i, distances, radprof, fwhm, plsc, max_adi_snr,
                  wavelengths=None):
    """
    """
    d = distances[i]
    snr = 0.01
    flux = radprof[i]
    snrs = []
    counter = 1

    while snr < 1.2 * max_adi_snr:
        f, snr = _get_adi_snrs(GARRPSF, GARRPA, fwhm, plsc, (flux, d, 0),
                               wavelengths)
        if counter > 2 and snr <= snrs[-1]:
            break
        snrs.append(snr)
        flux *= 2
        counter += 1
    return flux


def _get_adi_snrs(psf, angle_list, fwhm, plsc, flux_dist_theta_all,
                  wavelengths=None):
    """ Get the mean S/N (at 3 equidistant positions) for a given flux and
    distance, on a median subtracted frame.
    """
    snrs = []
    theta = flux_dist_theta_all[2]
    flux = flux_dist_theta_all[0]
    dist = flux_dist_theta_all[1]

    # 3 equidistant azimuthal positions
    for ang in [theta, theta + 120, theta + 240]:
        cube_fc, cx, cy = create_synt_cube(GARRAY, psf, angle_list, plsc,
                                           flux=flux, dist=dist, theta=ang,
                                           verbose=False)
        fr_temp = median_sub(cube_fc, angle_list, scale_list=wavelengths,
                             verbose=False)
        res = frame_quick_report(fr_temp, fwhm, source_xy=(cx, cy),
                                 verbose=False)
        # mean S/N in circular aperture
        snrs.append(np.mean(res[-1]))

    # median of mean S/N at 3 equidistant positions
    median_snr = np.median(snrs)
    return flux, median_snr


def _sample_flux_snr(distances, fwhm, plsc, n_injections, flux_min, flux_max,
                     nproc=10, random_seed=42, wavelengths=None):
    """
    Sensible flux intervals depend on a combination of factors, # of frames,
    range of rotation, correlation, glare intensity.
    """
    starttime = time_ini()
    if GARRAY.ndim == 3:
        frsize = int(GARRAY.shape[1])
    elif GARRAY.ndim == 4:
        frsize = int(GARRAY.shape[2])
    ninj = n_injections
    random_state = np.random.RandomState(random_seed)
    flux_dist_theta_all = list()
    snrs_list = list()
    fluxes_list = list()

    for i, d in enumerate(distances):
        yy, xx = get_annulus_segments((frsize, frsize), d, 1, 1)[0]
        num_patches = yy.shape[0]

        fluxes_dist = random_state.uniform(flux_min[i], flux_max[i], size=ninj)
        inds_inj = random_state.randint(0, num_patches, size=ninj)

        for j in range(ninj):
            injx = xx[inds_inj[j]]
            injy = yy[inds_inj[j]]
            injx -= frame_center(GARRAY[0])[1]
            injy -= frame_center(GARRAY[0])[0]
            dist = np.sqrt(injx ** 2 + injy ** 2)
            theta = np.mod(np.arctan2(injy, injx) / np.pi * 180, 360)
            flux_dist_theta_all.append((fluxes_dist[j], dist, theta))

    # Multiprocessing pool
    res = pool_map(nproc, _get_adi_snrs, GARRPSF, GARRPA, fwhm, plsc,
                   fixed(flux_dist_theta_all), wavelengths)

    for i in range(len(distances)):
        flux_dist = []
        snr_dist = []
        for j in range(ninj):
            flux_dist.append(res[j + (ninj * i)][0])
            snr_dist.append(res[j + (ninj * i)][1])
        fluxes_list.append(flux_dist)
        snrs_list.append(snr_dist)

    timing(starttime)
    return fluxes_list, snrs_list


def create_synt_cube(cube, psf, ang, plsc, dist=None, theta=None, flux=None,
                     random_seed=42, verbose=False):
    """
    """
    centy_fr, centx_fr = frame_center(cube[0])
    random_state = np.random.RandomState(random_seed)
    if theta is None:
        theta = random_state.randint(0,360)

    posy = dist * np.sin(np.deg2rad(theta)) + centy_fr
    posx = dist * np.cos(np.deg2rad(theta)) + centx_fr
    if verbose:
        print('Theta:', theta)
        print('Flux_inj:', flux)
    cubefc = cube_inject_companions(cube, psf, ang, flevel=flux, plsc=plsc,
                                    rad_dists=[dist], n_branches=1, theta=theta,
                                    verbose=verbose)

    return cubefc, posx, posy