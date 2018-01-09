
import os
import matplotlib.pyplot as plt
from raysect.optical import World

from cherab.aug.machine import plot_aug_wall_outline
import cherab.aug.bolometry
from cherab.aug.bolometry import load_default_bolometer_config
from cherab.aug.bolometry import load_standard_inversion_grid

plt.ion()

grid = load_standard_inversion_grid()
full_world = World()


def plot_detector(detector):
    detector.los_radiance_sensitivity.plot()
    plot_aug_wall_outline()
    detector.volume_radiance_sensitivity.plot()
    plot_aug_wall_outline()


flx = load_default_bolometer_config('FLX', parent=full_world, inversion_grid=grid)
for detector in flx:
    plot_detector(detector)
plt.show()
input("waiting...")
plt.close('all')

fdc = load_default_bolometer_config('FDC', parent=full_world, inversion_grid=grid)
for detector in fdc:
    plot_detector(detector)
plt.show()
input("waiting...")
plt.close('all')

fvc = load_default_bolometer_config('FVC', parent=full_world, inversion_grid=grid)
for detector in fvc:
    plot_detector(detector)
plt.show()
input("waiting...")
plt.close('all')

fhs = load_default_bolometer_config('FHS', parent=full_world, inversion_grid=grid)
for detector in fhs:
    plot_detector(detector)
plt.show()
input("waiting...")
plt.close('all')

fhc = load_default_bolometer_config('FHC', parent=full_world, inversion_grid=grid)
for detector in fhc:
    plot_detector(detector)
plt.show()
input("waiting...")
plt.close('all')

flh = load_default_bolometer_config('FLH', parent=full_world, inversion_grid=grid)
for detector in flh:
    plot_detector(detector)
plt.show()
input("waiting...")
plt.close('all')
