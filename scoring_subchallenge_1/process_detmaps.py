import sys
import os
import os.path
import numpy as np
import vip_hci as vip
import hciplot as hp
from vip_hci.fits import open_fits
from vip_hci.metrics import compute_binary_map


def process_detmaps(list_instruments, list_nb_datasets, input_dir, output_dir,
                    subchallenge, debug=False):
    """
    Processes the detection maps for a given list of instruments and
    number of DSs per instrument. Writes to a txt file the computed
    metrics.

    ADI
    list_instruments = ["sphere_irdis", "nirc2", "lmircam"]
    list_nb_datasets = [3, 3, 3]

    ADI + mSDI
    list_instruments = ["gpi", "sphere_ifs"]
    list_nb_datasets = [5, 5]
    """
    submit_dir = os.path.join(input_dir, 'res')
    truth_dir = os.path.join(input_dir, 'ref')

    if not os.path.isdir(submit_dir):
        print("%s doesn't exist" % submit_dir)

    if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_filename = os.path.join(output_dir, 'scores.txt')
        output_file = open(output_filename, 'w')

        threshold = open_fits(os.path.join(submit_dir, "detection_threshold"),
                              verbose=False)
        print("Global detection threshold: {}".format(threshold))

        # optional files
        npix_file = os.path.join(submit_dir, "npix.fits")
        overlap_threshold_file = os.path.join(submit_dir, "overlap_threshold")
        max_blob_fact_file = os.path.join(submit_dir, "max_blob_fact")

        if os.path.exists(npix_file):
            npix = open_fits(npix_file, verbose=False)
        else:
            npix = 2

        if os.path.exists(overlap_threshold_file):
            overlap_thr = open_fits(overlap_threshold_file, verbose=False)
        else:
            overlap_thr = 0.7
        if os.path.exists(max_blob_fact_file):
            max_blob_fact = open_fits(max_blob_fact_file, verbose=False)
        else:
            max_blob_fact = 2

        n_det_tot = 0.
        n_fps_tot = 0.
        n_det_truth = 0.

        for instru, nb_ds in zip(list_instruments, list_nb_datasets):
            for i in range(nb_ds):
                # this is a list/tuple of (x,y) positions
                truth_file = os.path.join(truth_dir,
                                          instru + "_positions_" + str(i + 1))
                submission_file = os.path.join(submit_dir,
                                               instru + "_detmap_" + str(i + 1))
                detection_map = open_fits(submission_file, verbose=False)

                if not os.path.exists(submission_file + ".fits"):
                    errmsg = "Detection map ({}) is missing"
                    raise ValueError(errmsg.format(submission_file + ".fits"))

                # truth_file e.g. np.array([(26.86, 22.42), (62,63)])
                injections = open_fits(truth_file, verbose=False)
                print("Instrument: {}, id: {}".format(instru, i + 1))
                if len(injections) != 0:
                    print("Injections: ", injections)

                    if debug:
                        hp.plot_frames(detection_map, title='Detection map')

                    injections = tuple((inj[0], inj[1]) for inj in injections)

                    fwhm_file = os.path.join(submit_dir, instru + "_fwhm_" +
                                             str(i + 1))
                    if os.path.exists(fwhm_file + ".fits"):
                        fwhm = float(open_fits(fwhm_file, verbose=False))
                    else:
                        fwhm_file = os.path.join(truth_dir, instru + "_fwhm_" +
                                                 str(i + 1))
                        fwhm = float(open_fits(fwhm_file, verbose=False))

                    res = compute_binary_map(detection_map,
                                             thresholds=[threshold],
                                             injections=injections,
                                             fwhm=fwhm, npix=npix,
                                             overlap_threshold=overlap_thr,
                                             max_blob_fact=max_blob_fact,
                                             debug=False, plot=debug)
                    n_det, n_fps, binmaps = res

                n_det_tot += n_det[0]
                n_fps_tot += n_fps[0]
                n_det_truth += len(injections)

        # Computing the metrics and saving to disk
        if n_det_truth > 0:
            recall = n_det_tot / n_det_truth
        else:
            recall = 0
        if n_fps_tot + n_det_tot != 0:
            prec = n_det_tot / (n_fps_tot + n_det_tot)
            FDR = n_fps_tot / (n_fps_tot + n_det_tot)
            f1 = 2 * prec * recall / (prec + recall)
        else:
            prec = 0
            FDR = 0
            f1 = 0

        # This file is read by Codalab to create the scoreboards
        # Key (in Codalab) is one of the following:
        # {'FalsePositives_xx', 'Precision_xx', 'F1_xx', 'Recall_xx'}
        if subchallenge == '3d':
            output_file.write("F1_3d: %f \n" % f1)
            output_file.write("Precision_3d: %f \n" % prec)
            output_file.write("Recall_3d: %f \n" % recall)
            output_file.write("FalsePositives_3d: %f \n" % n_fps_tot)
            output_file.close()
        elif subchallenge == '4d':
            output_file.write("F1_4d: %f \n" % f1)
            output_file.write("Precision_4d: %f \n" % prec)
            output_file.write("Recall_4d: %f \n" % recall)
            output_file.write("FalsePositives_4d: %f \n" % n_fps_tot)
            output_file.close()
        else:
            raise ValueError("`subchallenge` must be 3d or 4d")

        if debug:
            print("TPR (recall or sensitivity): %f \n" % recall)
            print("False Positives: %f \n" % n_fps_tot)
            print("FDR: %f \n" % FDR)
            print("Precision: %f \n" % prec)
            print("F1: %f \n" % f1)

