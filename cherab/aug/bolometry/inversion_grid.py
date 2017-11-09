
import numpy as np
import os
from cherab.tools.observers.inversion_grid import load_inversion_grid


def load_standard_inversion_grid():

    directory = os.path.split(__file__)[0]

    return load_inversion_grid(os.path.join(directory, "standard_grid.json"))


# A boolean mask array. Helps map cells in the original 2D AUG grid to the 1D CHERAB grid.
mask_file = os.path.join(os.path.split(__file__)[0], 'grid_mask.ndarray')
AUG_2D_TO_CHERAB_1D_GRID_MASK = np.load(open(mask_file, 'rb'))
