
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from cherab.tools.observers.bolometry import assemble_weight_matrix
from cherab.tools.observers.inversion_grid import EmissivityGrid
from cherab.aug.bolometry import load_standard_inversion_grid, load_default_bolometer_config
from cherab.aug.machine import plot_aug_wall_outline


# 'FVC1_A_CH1', 'FVC2_E_CH17', 'FLX_A_CH1', 'FHS_A_CH1'
EXCLUDED_CHANNELS = ['FDC_A_CH1', 'FDC_G_CH28', 'FHC1_A_CH1', 'FHC1_A_CH2', 'FHC1_A_CH3']

grid = load_standard_inversion_grid()

fhc = load_default_bolometer_config('FHC', inversion_grid=grid)
flh = load_default_bolometer_config('FLH', inversion_grid=grid)
fhs = load_default_bolometer_config('FHS', inversion_grid=grid)
fvc = load_default_bolometer_config('FVC', inversion_grid=grid)
fdc = load_default_bolometer_config('FDC', inversion_grid=grid)
flx = load_default_bolometer_config('FLX', inversion_grid=grid)


detector_keys, los_weight_matrix, vol_weight_matrix = assemble_weight_matrix([fhc, flh, fhs, fvc, fdc, flx],
                                                                             excluded_detectors=EXCLUDED_CHANNELS)

num_detectors, num_sources = los_weight_matrix.shape

los_projected = np.zeros(num_sources)
vol_projected = np.zeros(num_sources)
los_projected_norm = np.zeros(num_sources)
vol_projected_norm = np.zeros(num_sources)

for i in range(num_detectors):

    los_projected += los_weight_matrix[i, :]
    vol_projected += vol_weight_matrix[i, :]

    los_projected_norm += los_weight_matrix[i, :] / los_weight_matrix[i, :].sum()
    vol_projected_norm += vol_weight_matrix[i, :] / vol_weight_matrix[i, :].sum()


# inverted_emiss = EmissivityGrid(grid, case_id='LOS projected', emissivities=los_projected)
# inverted_emiss.plot()
# plt.axis('equal')
#
# inverted_emiss = EmissivityGrid(grid, case_id='Vol projected', emissivities=vol_projected)
# inverted_emiss.plot()
# plt.axis('equal')

plt.ion()

patches = []
for i in range(num_sources):
    polygon = Polygon(grid.cell_data[i], True)
    patches.append(polygon)

p = PatchCollection(patches)
p.set_array(los_projected_norm)
p.set_clim([0, 0.25])

fig, ax = plt.subplots()
ax.add_collection(p)
plt.xlim(1, 2.5)
plt.ylim(-1.5, 1.5)
plt.title('LOS projected + normalised')
plot_aug_wall_outline()
plt.axis('equal')


patches = []
for i in range(num_sources):
    polygon = Polygon(grid.cell_data[i], True)
    patches.append(polygon)

p = PatchCollection(patches)
p.set_array(vol_projected_norm)
p.set_clim([0, 0.25])

fig, ax = plt.subplots()
ax.add_collection(p)
plt.xlim(1, 2.5)
plt.ylim(-1.5, 1.5)
plt.title('Vol projected + normalised')
plot_aug_wall_outline()
plt.axis('equal')
