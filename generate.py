import numpy as np
import vip_hci as vip
import os
import glob
from vip_hci.var import fit_2dgaussian
from flux_estimation import estimate_fluxes

## Fake data
def generate_fake_cubes(instruments=["sphere_irdis","nirc2"],N=2,orig_dir="fakecubes",data_dir="public_data"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    cube = vip.fits.open_fits('./data_sample/naco_betapic_cube.fits') 
    pa = vip.fits.open_fits('./data_sample/naco_betapic_pa.fits')
    psf = vip.fits.open_fits('./data_sample/naco_betapic_psf.fits')
    plsc = vip.conf.VLT_NACO['plsc']
    vip.var.fit_2dgaussian(psf, crop=11, full_output=True, debug=False)
    #fwhm = np.mean((4.92, 4.67))
    bounds_flux = np.array([50, 90])
    for instru in instruments:
        for i in range(N):
            vip.fits.write_fits(os.path.join(orig_dir,instru+"_cube_clean_"+str(i+1)+".fits"),cube)
            #vip.fits.write_fits(os.path.join(data_dir,instru+"_fwhm_"+str(i+1)+".fits"),np.array([fwhm]))
            vip.fits.write_fits(os.path.join(data_dir,instru+"_pa_"+str(i+1)+".fits"),pa)
            vip.fits.write_fits(os.path.join(data_dir,instru+"_psf_"+str(i+1)+".fits"),psf)
            vip.fits.write_fits(os.path.join(data_dir,instru+"_plsc_"+str(i+1)+".fits"),np.array([plsc]))
            


def inject_companions(instruments=["sphere_irdis","nirc2"],orig_dir="fakecubes",data_dir="public_data",ref_dir="references",fourD=False,listNbDS=[1]):
    listFlux = dict()
    listPos = dict()
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(ref_dir):
        os.makedirs(ref_dir)
    for instru,N in zip(instruments,listNbDS):
        listFlux[instru]=[]
        listPos[instru]=[]
        for i in range(N):
            listPos[instru].append([])
            listFlux[instru].append([])
            # load data
            cube = vip.fits.open_fits(os.path.join(orig_dir,instru+"_cube_clean_"+str(i+1)+".fits"))
            psf = vip.fits.open_fits(os.path.join(data_dir,instru+"_psf_"+str(i+1)+".fits"))
            pa = vip.fits.open_fits(os.path.join(data_dir,instru+"_pa_"+str(i+1)+".fits"))
            plsc = vip.fits.open_fits(os.path.join(data_dir,instru+"_plsc_"+str(i+1)+".fits"))[0]            
            
            if fourD is True:
                fwhm = np.mean([fit_2dgaussian(np.mean(psf, axis=0), full_output=True)['fwhm_x'][0],
                fit_2dgaussian(np.mean(psf, axis=0), full_output=True)['fwhm_y'][0]])
                wl = vip.fits.open_fits(os.path.join(orig_dir,instru+"_wl_"+str(i+1)+".fits"))
                
            else:
                fwhm = np.mean([fit_2dgaussian(psf, full_output=True)['fwhm_x'][0],
                fit_2dgaussian(psf, full_output=True)['fwhm_y'][0]])
                

            nb_inj = np.random.randint(0,6)
            
            max_steps_fwhm = int(np.floor(np.minimum(cube.shape[-1],cube.shape[-2])/fwhm))
            if fourD is True:
                bounds_flux_hard = estimate_fluxes(cube, psf, (np.arange(3,max_steps_fwhm)*fwhm).astype(int), pa, fwhm, plsc, wavelengths=wl,min_adi_snr=1, max_adi_snr=3)
                bounds_flux_soft = estimate_fluxes(cube, psf, (np.arange(3,max_steps_fwhm)*fwhm).astype(int), pa, fwhm, plsc, wavelengths=wl,min_adi_snr=3, max_adi_snr=5)
            else:
                bounds_flux_hard = estimate_fluxes(cube, psf, (np.arange(3,max_steps_fwhm)*fwhm).astype(int), pa, fwhm, plsc, wavelengths=None,min_adi_snr=1, max_adi_snr=3)
                bounds_flux_soft = estimate_fluxes(cube, psf, (np.arange(3,max_steps_fwhm)*fwhm).astype(int), pa, fwhm, plsc, wavelengths=None,min_adi_snr=3, max_adi_snr=5)
            
            cubefc=cube #init
            positions=[]
            for j in range(nb_inj):
                regime = np.random.randint(0,2)
                if regime == 0:
                    rad_dists=np.random.uniform(2.5*fwhm,6*fwhm)
                else:
                    rad_dists=np.random.uniform(6*fwhm,20*fwhm)
                rad_dists = np.min([rad_dists,cube.shape[1]/2-2*fwhm,cube.shape[2]/2-2*fwhm]) #avoid going outside of the cube
                theta = np.random.uniform(0,360)
                
                dist_ind = np.minimum(np.maximum(int(rad_dists/fwhm),3),max_steps_fwhm)-3 # find closest distance index for in bound_fluxes
                
                if j%2==0: #one of two is drawn as difficult
                    flux = np.random.uniform(bounds_flux_hard[0][dist_ind],bounds_flux_hard[1][dist_ind])
                else:
                    flux = np.random.uniform(bounds_flux_soft[0][dist_ind],bounds_flux_soft[1][dist_ind])
                if fourD:
                    flux = np.ones_like(wl)*flux
                
                listFlux[instru][i].append(flux)
                
                cubefc,positions = vip.metrics.cube_inject_companions(cubefc, psf, pa, flux,  plsc, rad_dists, 1, theta,full_output=True,verbose=False)
                listPos[instru][i] += positions
                
            vip.fits.write_fits(os.path.join(data_dir,instru+"_cube_"+str(i+1)+".fits"),cubefc)
            vip.fits.write_fits(os.path.join(data_dir,instru+"_fwhm_"+str(i+1)+".fits"),np.array([fwhm]))
 
            vip.fits.write_fits(os.path.join(ref_dir,instru+"_positions_"+str(i+1)+".fits"),np.array(listPos[instru][i]))
            vip.fits.write_fits(os.path.join(data_dir,instru+"_bounds_"+str(i+1)+".fits"),np.array([bounds_flux_hard[0],bounds_flux_soft[1]]))    
            vip.fits.write_fits(os.path.join(ref_dir,instru+"_flux_"+str(i+1)+".fits"),np.array(listFlux[instru][i]))
                        
                
    return listPos,listFlux