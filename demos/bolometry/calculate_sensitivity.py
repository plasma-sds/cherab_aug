
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from raysect.optical import World

from cherab.aug.machine import plot_aug_wall_outline, import_mesh_segment, VESSEL, PSL, ICRH, DIVERTOR, A_B_COILS
from cherab.aug.bolometry import FDC_TUBE, FLX_TUBE, FVC_TUBE, FHS_TUBE, load_default_bolometer_config
from cherab.aug.bolometry import load_standard_inversion_grid


grid = load_standard_inversion_grid()

flx_world = World()
import_mesh_segment(flx_world, FLX_TUBE)
flx = load_default_bolometer_config('FLX', parent=flx_world)

flx_a_ch4 = flx['FLX_A_CH4']


flx_a_ch4.calculate_sensitivity(grid, flx_world)

los_sensitivity_grid = flx_a_ch4.los_sensitivity
patches = []

for i in range(los_sensitivity_grid.count):
    polygon = Polygon(los_sensitivity_grid.grid_geometry.cell_data[i], True)
    patches.append(polygon)

p = PatchCollection(patches)
p.set_array(los_sensitivity_grid.sensitivity)

fig, ax = plt.subplots()
ax.add_collection(p)
plt.xlim(1, 2.5)
plt.ylim(-1.5, 1.5)
plt.title("Line of sight sensitivity")


vol_sensitivity_grid = flx_a_ch4.volume_sensitivity
patches = []

for i in range(vol_sensitivity_grid.count):
    polygon = Polygon(vol_sensitivity_grid.grid_geometry.cell_data[i], True)
    patches.append(polygon)

p = PatchCollection(patches)
p.set_array(vol_sensitivity_grid.sensitivity)

fig, ax = plt.subplots()
ax.add_collection(p)
plt.xlim(1, 2.5)
plt.ylim(-1.5, 1.5)
plt.title("Line of sight sensitivity")
plt.axis('equal')

plt.ion()
plt.show()
plt.ioff()
