
import os
import time
import matplotlib.pyplot as plt
from raysect.optical import World

from cherab.aug.machine import plot_aug_wall_outline, import_mesh_segment, VESSEL, PSL, ICRH, DIVERTOR, A_B_COILS
import cherab.aug.bolometry
from cherab.aug.bolometry import FDC_TUBE, FLX_TUBE, FVC_TUBE, FHS_TUBE, load_default_bolometer_config
from cherab.aug.bolometry import load_standard_inversion_grid


start_time = time.time()

# Load default inversion grid
grid = load_standard_inversion_grid()


# Calculate FLX camera sensitivities
flx_world = World()
import_mesh_segment(flx_world, FLX_TUBE)
flx = load_default_bolometer_config('FLX', parent=flx_world)
for detector in flx:
    print('calculating detector {}'.format(detector.detector_id))
    detector.calculate_sensitivity(grid)
    detector.save_sensitivities(dir_path=os.path.split(cherab.aug.bolometry.detectors.__file__)[0])


# Calculate FDC camera sensitivities
fdc_world = World()
import_mesh_segment(fdc_world, FDC_TUBE)
fdc = load_default_bolometer_config('FDC', parent=fdc_world)
for detector in fdc:
    print('calculating detector {}'.format(detector.detector_id))
    detector.calculate_sensitivity(grid)
    detector.save_sensitivities(dir_path=os.path.split(cherab.aug.bolometry.detectors.__file__)[0])


# Calculate FVC camera sensitivities
fvc_world = World()
import_mesh_segment(fvc_world, FVC_TUBE)
fvc = load_default_bolometer_config('FVC', parent=fvc_world)
for detector in fvc:
    print('calculating detector {}'.format(detector.detector_id))
    detector.calculate_sensitivity(grid)
    detector.save_sensitivities(dir_path=os.path.split(cherab.aug.bolometry.detectors.__file__)[0])


# Calculate FVC camera sensitivities
fvc_world = World()
import_mesh_segment(fvc_world, FVC_TUBE)
fvc = load_default_bolometer_config('FVC', parent=fvc_world)
for detector in fvc:
    print('calculating detector {}'.format(detector.detector_id))
    detector.calculate_sensitivity(grid)
    detector.save_sensitivities(dir_path=os.path.split(cherab.aug.bolometry.detectors.__file__)[0])


# Calculate FHS camera sensitivities
fhs_world = World()
import_mesh_segment(fhs_world, FHS_TUBE)
fhs = load_default_bolometer_config('FHS', parent=fhs_world)
for detector in fhs:
    print('calculating detector {}'.format(detector.detector_id))
    detector.calculate_sensitivity(grid)
    detector.save_sensitivities(dir_path=os.path.split(cherab.aug.bolometry.detectors.__file__)[0])


# FHC and FLH use the full AUG mesh structure
full_world = World()
FULL_MESH_SEGMENTS = VESSEL + PSL + ICRH + DIVERTOR + A_B_COILS
import_mesh_segment(full_world, FULL_MESH_SEGMENTS)

# Calculate FHC camera sensitivities
fhc = load_default_bolometer_config('FHC', parent=full_world)
for detector in fhc:
    print('calculating detector {}'.format(detector.detector_id))
    detector.calculate_sensitivity(grid)
    detector.save_sensitivities(dir_path=os.path.split(cherab.aug.bolometry.detectors.__file__)[0])

# Calculate FLH camera sensitivities
flh = load_default_bolometer_config('FLH', parent=full_world)
for detector in flh:
    print('calculating detector {}'.format(detector.detector_id))
    detector.calculate_sensitivity(grid)
    detector.save_sensitivities(dir_path=os.path.split(cherab.aug.bolometry.detectors.__file__)[0])

print("run time - {:.2G}mins".format((time.time() - start_time) / 60))
