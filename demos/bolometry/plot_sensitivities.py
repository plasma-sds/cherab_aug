
import os
import matplotlib.pyplot as plt
from raysect.optical import World

from cherab.aug.machine import plot_aug_wall_outline, import_mesh_segment
import cherab.aug.bolometry
from cherab.aug.bolometry import FDC_TUBE, FLX_TUBE, FVC_TUBE, FHS_TUBE, load_default_bolometer_config
from cherab.aug.bolometry import load_standard_inversion_grid

plt.ion()

grid = load_standard_inversion_grid()

flx_world = World()
import_mesh_segment(flx_world, FLX_TUBE)
flx = load_default_bolometer_config('FLX', parent=flx_world)

for detector in flx:
    filename = "{}_sensitivity.pickle".format(detector.detector_id)
    file_path = os.path.join(os.path.split(cherab.aug.bolometry.detectors.__file__)[0], filename)
    detector.reload_sensitivity(file_path, grid)

    detector.los_sensitivity.plot()
    plot_aug_wall_outline()

    detector.volume_sensitivity.plot()
    plot_aug_wall_outline()

plt.show()
plt.ioff()
