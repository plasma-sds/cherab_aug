
import os
from cherab.tools.observers.inversion_grid import load_inversion_grid


def load_standard_inversion_grid():

    directory = os.path.split(__file__)[0]

    return load_inversion_grid(os.path.join(directory, "standard_grid.json"))
