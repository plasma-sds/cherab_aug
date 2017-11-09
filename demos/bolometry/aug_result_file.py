
import os
import matplotlib.pyplot as plt

from cherab.aug.bolometry import load_standard_inversion_grid, load_blb_config_file, load_blb_result_file
from cherab.aug.machine import plot_aug_wall_outline


# replace with path to your BLB geometry directory
AUG_BLB_CASE_DIRECTORY = os.path.expanduser("~/CCFE/mst1/aug_bolometry/BLB.33280_2")

GRID_DIMENSIONS = "^Grid points:\s*([0-9]+)x([0-9]+)$"
USED_CAMERAS = "^Used cameras:\s*((?:[A-Z]{3}\s?)*)"

# Config file describing which bolometer cameras are active
grid = load_standard_inversion_grid()

active_cameras = load_blb_config_file(os.path.join(AUG_BLB_CASE_DIRECTORY, "config"))

result_blb33280 = load_blb_result_file(os.path.join(AUG_BLB_CASE_DIRECTORY, "resultef2d.BLB.33280_4.100000"), grid)


plt.ion()
result_blb33280.plot()
plot_aug_wall_outline()

plt.ioff()
plt.show()
