################################################################################
# ADI + mSDI (4D cubes)
################################################################################

import sys
import os
import os.path
from process_detmaps import process_detmaps

input_dir = sys.argv[1]
output_dir = sys.argv[2]
debug = sys.argv[3].lower() == 'true'

# Instruments
list_instruments = ["gpi", "sphere_ifs"]
list_nb_datasets = [5, 5]

process_detmaps(list_instruments, list_nb_datasets, input_dir, output_dir, '4d',
                debug)
