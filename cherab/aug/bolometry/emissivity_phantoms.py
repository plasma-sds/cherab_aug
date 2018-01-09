
import os
import pickle

from cherab.aug.bolometry import load_standard_inversion_grid
from cherab.tools.observers.inversion_grid import EmissivityGrid


grid = load_standard_inversion_grid()

dir_path = os.path.split(__file__)[0]
phantoms_file_path = os.path.join(dir_path, 'emissivity_phantoms.pickle')
fh = open(phantoms_file_path, 'rb')
EMISSIVITY_PHANTOMS = pickle.load(fh)
fh.close()


def load_emissivity_phantom(phantom_id):

    try:
        emissivity_values = EMISSIVITY_PHANTOMS[phantom_id]
    except IndexError:
        raise ValueError("Phantom ID '{}' was not found.".format(phantom_id))

    return EmissivityGrid(grid, case_id=phantom_id, emissivities=emissivity_values)
