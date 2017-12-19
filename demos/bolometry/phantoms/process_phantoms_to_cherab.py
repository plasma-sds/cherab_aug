
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

from cherab.tools.observers.inversion_grid import EmissivityGrid
from cherab.aug.bolometry import load_standard_inversion_grid

plt.ion()

grid = load_standard_inversion_grid()

PATH_TO_PHANTOMS = '/home/matt/CCFE/mst1/aug_bolometry/phantoms/emiss'

grid_mask = np.load('/home/matt/CCFE/cherab/aug/cherab/aug/bolometry/grid_mask.ndarray')


emissivity_phantoms = {}
for i in range(94):
    phantom_index = i + 1

    phantom_filename = str(phantom_index).zfill(3) + ".emiss"
    fh = open(os.path.join(PATH_TO_PHANTOMS, phantom_filename), 'r')
    lines = fh.readlines()
    lines = lines[3868:]
    assert len(lines) == 3864

    raw_values = np.array([float(line.strip()) for line in lines]).reshape((84, 46))

    mapped_phantom_values = np.zeros(grid.count)

    mapped_index = 0
    for i in range(83):
        for j in range(45):
            if grid_mask[i, j]:
                # need to convert cell values to W/m^3/str
                cell_avg = (raw_values[i, j] + raw_values[i+1, j] + raw_values[i+1, j+1] + raw_values[i, j+1]) / 4 / (4 * np.pi)
                mapped_phantom_values[mapped_index] = cell_avg
                mapped_index += 1

    phantom_id = "AUG_emission_phantom_" + str(phantom_index).zfill(3)
    emissivity_phantoms[phantom_id] = mapped_phantom_values

    emiss = EmissivityGrid(grid, case_id="phantom_id", emissivities=mapped_phantom_values)
    emiss.plot()
    plt.axis('equal')


pickle.dump(emissivity_phantoms, open('emissivity_phantoms.pickle', 'wb'))

