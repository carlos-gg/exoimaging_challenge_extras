################################################################################
# ADI (3D cubes)
################################################################################

import sys
import os
import os.path
from process_detmaps import process_detmaps

input_dir = sys.argv[1]
output_dir = sys.argv[2]
debug = sys.argv[3].lower() == 'true'

# Instruments
list_instruments = ["sphere_irdis", "nirc2", "lmircam"]
list_nb_datasets = [3, 3, 3]

process_detmaps(list_instruments, list_nb_datasets, input_dir, output_dir, '3d',
                debug)
