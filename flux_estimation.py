"""

"""
from __future__ import print_function, division, absolute_import

__all__ = ['EstimateFluxes']

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
from vip_hci.var import (cube_filter_highpass, pp_subplots,
                         get_annulus_segments, prepare_matrix)
from vip_hci.metrics import cube_inject_companions
from vip_hci.preproc import (check_pa_vector, cube_derotate, cube_crop_frames,
                             frame_rotate, frame_shift, frame_px_resampling,
                             frame_crop, cube_collapse, check_pa_vector,
                             check_scal_vector)
from vip_hci.preproc import cube_rescaling_wavelengths as scwave
from vip_hci.preproc.derotation import _compute_pa_thresh, _find_indices_adi
from vip_hci.metrics import frame_quick_report
from vip_hci.medsub import median_sub
from vip_hci.pca import pca, svd_wrapper


class EstimateFluxes:
    """
    Fluxes (proxy of contrast) estimator for injecting fake companions.
    """
    def __init__(self, cube, psf, distances, angles, fwhm, plsc,
                 wavelengths=None, n_injections=10, algo='median', n_comp=2,
                 scaling='temp-standard', min_snr=2, max_snr=5, random_seed=42,
                 n_proc=2):
        global GARRAY
        GARRAY = cube
        global GARRPSF
        GARRPSF = psf
        global GARRPA
        GARRPA = angles
        global GARRWL
        GARRWL = wavelengths
        self.min_fluxes = None
        self.max_fluxes = None
        self.radprof = None
        self.sampled_fluxes = None
        self.sampled_snrs = None
        self.estimated_fluxes_low = None
        self.estimated_fluxes_high = None
        self.distances = distances
        self.angles = angles
        self.fwhm = fwhm
        self.plsc = plsc
        self.scaling = scaling
        self.wavelengths = wavelengths
        self.n_injections = n_injections
        self.algo = algo
        self.n_comp = n_comp
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.random_seed = random_seed
        self.n_proc = n_proc
        self.n_dist = range(len(self.distances))

        if cube.ndim == 4:
            if wavelengths is None:
                raise ValueError('`wavelengths` parameter must be provided')

    def get_min_flux(self):
        """ Obtaining the low end of the interval for sampling the SNRs. Based
        on the initial estimation of the radial profile of the mean frame.
        """
        starttime = time_ini()

        # Getting the radial profile in the mean frame of the cube
        sampling_sep = 1
        radius_int = 1
        if GARRAY.ndim == 3:
            global_frame = np.mean(GARRAY, axis=0)
        elif GARRAY.ndim == 4:
            global_frame = np.mean(GARRAY.reshape(-1, GARRAY.shape[2],
                                                  GARRAY.shape[3]), axis=0)

        me = frame_average_radprofile(global_frame, sep=sampling_sep,
                                      init_rad=radius_int, plot=False)
        radprof = np.array(me.radprof)
        radprof = radprof[np.array(self.distances) + 1]
        radprof[radprof < 0] = 0.01
        self.radprof = radprof

        print("Estimating the min values for sampling the S/N vs flux function")
        flux_min = pool_map(self.n_proc, _get_min_flux, fixed(self.n_dist),
                            self.distances, radprof, self.fwhm, self.plsc,
                            self.min_snr, self.wavelengths, self.algo,
                            self.n_comp, self.scaling)

        self.min_fluxes = flux_min
        timing(starttime)

    def get_max_flux(self):
        """ Obtaining the high end of the interval for sampling the SNRs.
        """
        starttime = time_ini()

        print("Estimating the max values for sampling the S/N vs flux function")
        flux_max = pool_map(self.n_proc, _get_max_flux, fixed(self.n_dist),
                            self.distances, self.min_fluxes, self.fwhm,
                            self.plsc, self.max_snr, self.wavelengths,
                            self.algo, self.n_comp, self.scaling)

        self.max_fluxes = flux_max
        timing(starttime)

    def sampling(self):
        """ Using the computed interval of fluxes for sampling the flux vs SNR
        relationship.
        """
        if not self.min_fluxes:
            self.get_min_flux()

        if not self.max_fluxes:
            self.get_max_flux()

        starttime = time_ini()
        print("Sampling by injecting fake companions")
        res = _sample_flux_snr(self.distances, self.fwhm, self.plsc,
                               self.n_injections, self.min_fluxes,
                               self.max_fluxes, self.n_proc, self.random_seed,
                               self.wavelengths, self.algo, self.n_comp,
                               self.scaling)
        self.sampled_fluxes, self.sampled_snrs = res
        timing(starttime)

    def run(self, kernel='rbf', epsilon=0.1, c=1e4, gamma=1e-2, figsize=(10, 2),
            dpi=100, **kwargs):
        """ Building a regression model with he sampled fluxes and SNRs.

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
        if not self.sampled_fluxes or not self.sampled_snrs:
            self.sampling()

        starttime = time_ini()

        plotvlines = [self.min_snr, self.max_snr]
        nsubplots = len(self.distances)
        ncols = min(4, nsubplots)
        if nsubplots % 2 != 0:
            nsubplots -= 1
        nrows = int(nsubplots / ncols) + 1

        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi,
                                sharey='row')
        fig.subplots_adjust(wspace=0.05, hspace=0.3)
        axs = axs.ravel()
        fhi = list()
        flo = list()

        print("Building the regression models for each separation")
        # Regression for each distance
        for i, d in enumerate(self.distances):
            fluxes = np.array(self.sampled_fluxes[i])
            snrs = np.array(self.sampled_snrs[i])
            mask = np.where(snrs > 0.1)
            snrs = snrs[mask].reshape(-1, 1)
            fluxes = fluxes[mask].reshape(-1, 1)
            model = SVR(kernel=kernel, epsilon=epsilon, C=c, gamma=gamma,
                        **kwargs)
            model.fit(X=snrs, y=fluxes)
            flux_for_lowsnr = model.predict(self.min_snr)
            flux_for_higsnr = model.predict(self.max_snr)
            fhi.append(flux_for_higsnr[0])
            flo.append(flux_for_lowsnr[0])
            snrminp = self.min_snr / 2
            snrs_pred = np.linspace(snrminp, self.max_snr + snrminp,
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
            ax0.set_ylabel('Signal to noise ratio',
                           labelpad=25, size=8)

        for i in range(len(self.distances), len(axs)):
            axs[i].axis('off')

        flo = np.array(flo).flatten()
        fhi = np.array(fhi).flatten()
        self.estimated_fluxes_high = fhi
        self.estimated_fluxes_low = flo

        plt.figure(figsize=(10, 4), dpi=dpi)
        plt.plot(self.distances, self.radprof, '--', alpha=0.8, color='gray',
                 lw=2, label='average radial profile')
        plt.plot(self.distances, flo, '.-', alpha=0.6, lw=2, color='dodgerblue',
                 label='flux lower interval')
        plt.plot(self.distances, fhi, '.-', alpha=0.6, color='dodgerblue', lw=2,
                 label='flux upper interval')
        plt.fill_between(self.distances, flo, fhi, where=flo <= fhi, alpha=0.2,
                         facecolor='dodgerblue', interpolate=True)
        plt.grid(which='major', alpha=0.4)
        plt.xlabel('Distance from the center [Pixels]')
        plt.ylabel('Fakecomp flux scaling [Counts]')
        plt.minorticks_on()
        plt.xlim(0)
        plt.ylim(0)
        plt.legend()
        plt.show()
        timing(starttime)


def _get_min_flux(i, distances, radprof, fwhm, plsc, min_snr, wavelengths=None,
                  mode='pca', ncomp=2, scaling='temp-standard'):
    """
    """
    d = distances[i]
    fmin = radprof[i] * 0.1
    _, snr = _get_adi_snrs(GARRPSF, GARRPA, fwhm, plsc, (fmin, d, 0),
                           wavelengths, mode, ncomp, scaling)

    while snr > min_snr:
        f, snr = _get_adi_snrs(GARRPSF, GARRPA, fwhm, plsc, (fmin, d, 0),
                               wavelengths, mode, ncomp, scaling)
        fmin *= 0.5

    return fmin


def _get_max_flux(i, distances, flux_min, fwhm, plsc, max_snr, wavelengths=None,
                  mode='pca', ncomp=2, scaling='temp-standard'):
    """
    """
    d = distances[i]
    snr = 0.01
    flux = flux_min[i]
    snrs = []
    counter = 1

    while snr < 1.2 * max_snr:
        f, snr = _get_adi_snrs(GARRPSF, GARRPA, fwhm, plsc, (flux, d, 0),
                               wavelengths, mode, ncomp, scaling)

        # checking that the snr does not decrease
        if counter > 2 and snr <= snrs[-1] and snr > 1.2 * max_snr:
            break

        snrs.append(snr)
        flux *= 1.5
        counter += 1
    return flux


def _sample_flux_snr(distances, fwhm, plsc, n_injections, flux_min, flux_max,
                     nproc=10, random_seed=42, wavelengths=None, mode='median',
                     ncomp=2, scaling='temp-standard'):
    """
    Sensible flux intervals depend on a combination of factors, # of frames,
    range of rotation, correlation, glare intensity.
    """
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

    res = pool_map(nproc, _get_adi_snrs, GARRPSF, GARRPA, fwhm, plsc,
                   fixed(flux_dist_theta_all), wavelengths, mode, ncomp,
                   scaling)

    for i in range(len(distances)):
        flux_dist = []
        snr_dist = []
        for j in range(ninj):
            flux_dist.append(res[j + (ninj * i)][0])
            snr_dist.append(res[j + (ninj * i)][1])
        fluxes_list.append(flux_dist)
        snrs_list.append(snr_dist)

    return fluxes_list, snrs_list


def _get_adi_snrs(psf, angle_list, fwhm, plsc, flux_dist_theta_all,
                  wavelengths=None, mode='median', ncomp=2,
                  scaling='temp-standard'):
    """ Get the mean S/N (at 3 equidistant positions) for a given flux and
    distance, on a median subtracted frame.
    """
    theta = flux_dist_theta_all[2]
    flux = flux_dist_theta_all[0]
    dist = flux_dist_theta_all[1]

    if mode == 'median':
        snrs = []
        # 3 equidistant azimuthal positions
        for ang in [theta, theta + 120, theta + 240]:
            cube_fc, posx, posy = create_synt_cube(GARRAY, psf, angle_list,
                                                   plsc, flux=flux, dist=dist,
                                                   theta=ang, verbose=False)
            fr_temp = _compute_residual_frame(cube_fc, angle_list, dist, fwhm,
                                              wavelengths, mode, ncomp,
                                              'lapack', scaling,
                                              collapse='median', imlib='opencv',
                                              interpolation='lanczos4')
            res = frame_quick_report(fr_temp, fwhm, source_xy=(posx, posy),
                                     verbose=False)
            # mean S/N in circular aperture
            snrs.append(np.mean(res[-1]))

        # median of mean S/N at 3 equidistant positions
        snr = np.median(snrs)

    elif mode == 'pca':
        cube_fc, posx, posy = create_synt_cube(GARRAY, psf, angle_list, plsc,
                                               flux=flux, dist=dist,
                                               theta=theta, verbose=False)
        fr_temp = _compute_residual_frame(cube_fc, angle_list, dist, fwhm,
                                          wavelengths, mode, ncomp,
                                          'lapack', scaling, collapse='median',
                                          imlib='opencv',
                                          interpolation='lanczos4')
        res = frame_quick_report(fr_temp, fwhm, source_xy=(posx, posy),
                                 verbose=False)
        # mean S/N in circular aperture
        snr = np.mean(res[-1])

    return flux, snr


def _compute_residual_frame(cube, angle_list, radius, fwhm, wavelengths=None,
                            mode='pca', ncomp=2, svd_mode='lapack',
                            scaling='temp-standard', collapse='median',
                            imlib='opencv', interpolation='lanczos4'):
    """
    """
    annulus_width = 2 * fwhm

    if cube.ndim == 3:
        if mode == 'pca':
            angle_list = check_pa_vector(angle_list)
            data, ind = prepare_matrix(cube, scaling, mode='annular',
                                       annulus_radius=radius, verbose=False,
                                       annulus_width=annulus_width)
            yy, xx = ind
            V = svd_wrapper(data, svd_mode, ncomp, False, False)
            transformed = np.dot(V, data.T)
            reconstructed = np.dot(transformed.T, V)
            residuals = data - reconstructed
            cube_empty = np.zeros_like(cube)
            cube_empty[:, yy, xx] = residuals
            cube_res_der = cube_derotate(cube_empty, angle_list, imlib=imlib,
                                         interpolation=interpolation)
            res_frame = cube_collapse(cube_res_der, mode=collapse)

        elif mode == 'median':
            res_frame = median_sub(cube, angle_list, verbose=False)

    elif cube.ndim == 4:
        if mode == 'pca':
            z, n, y_in, x_in = cube.shape
            angle_list = check_pa_vector(angle_list)
            scale_list = check_scal_vector(wavelengths)
            big_cube = []

            # Rescaling the spectral channels to align the speckles
            for i in range(n):
                cube_resc = scwave(cube[:, i, :, :], scale_list)[0]
                cube_resc = cube_crop_frames(cube_resc, size=y_in,
                                             verbose=False)
                big_cube.append(cube_resc)

            big_cube = np.array(big_cube)
            big_cube = big_cube.reshape(z * n, y_in, x_in)

            data, ind = prepare_matrix(big_cube, scaling, mode='annular',
                                       annulus_radius=radius, verbose=False,
                                       annulus_width=annulus_width)
            yy, xx = ind
            V = svd_wrapper(data, svd_mode, ncomp, False, False)
            transformed = np.dot(V, data.T)
            reconstructed = np.dot(transformed.T, V)
            residuals = data - reconstructed
            res_cube = np.zeros_like(big_cube)
            res_cube[:, yy, xx] = residuals

            # Descaling the spectral channels
            resadi_cube = np.zeros((n, y_in, x_in))
            for i in range(n):
                frame_i = scwave(res_cube[i * z:(i + 1) * z, :, :], scale_list,
                                 full_output=False, inverse=True, y_in=y_in,
                                 x_in=x_in, collapse=collapse)
                resadi_cube[i] = frame_i

            cube_res_der = cube_derotate(resadi_cube, angle_list, imlib=imlib,
                                         interpolation=interpolation)
            res_frame = cube_collapse(cube_res_der, mode=collapse)

        elif mode == 'median':
            res_frame = median_sub(cube, angle_list, scale_list=wavelengths,
                                   verbose=False)

    return res_frame


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