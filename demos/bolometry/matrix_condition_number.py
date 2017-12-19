
import numpy as np
import matplotlib.pyplot as plt

from cherab.tools.observers.inversion_grid import SensitivityMatrix
from cherab.aug.bolometry import load_standard_inversion_grid
from cherab.aug.bolometry import load_default_bolometer_config


EXCLUDED_CHANNELS = ['FVC1_A_CH1', 'FVC2_E_CH17', 'FLX_A_CH1', 'FDC_A_CH1', 'FDC_G_CH28', 'FHS_A_CH1',
                     'FHC1_A_CH1', 'FHC1_A_CH2', 'FHC1_A_CH3']

grid = load_standard_inversion_grid()


fhc = load_default_bolometer_config('FHC', inversion_grid=grid)
flh = load_default_bolometer_config('FLH', inversion_grid=grid)
fhs = load_default_bolometer_config('FHS', inversion_grid=grid)
fvc = load_default_bolometer_config('FVC', inversion_grid=grid)
fdc = load_default_bolometer_config('FDC', inversion_grid=grid)
flx = load_default_bolometer_config('FLX', inversion_grid=grid)

num_detectors = 0
for detector in fhc:
    if detector.detector_id not in EXCLUDED_CHANNELS:
        num_detectors += 1
for detector in flh:
    if detector.detector_id not in EXCLUDED_CHANNELS:
        num_detectors += 1
for detector in fhs:
    if detector.detector_id not in EXCLUDED_CHANNELS:
        num_detectors += 1
for detector in fvc:
    if detector.detector_id not in EXCLUDED_CHANNELS:
        num_detectors += 1
for detector in fdc:
    if detector.detector_id not in EXCLUDED_CHANNELS:
        num_detectors += 1
for detector in flx:
    if detector.detector_id not in EXCLUDED_CHANNELS:
        num_detectors += 1

num_sensitivities = len(detector._volume_radiance_sensitivity.sensitivity)

los_weight_matrix = np.zeros((num_detectors, num_sensitivities))
vol_weight_matrix = np.zeros((num_detectors, num_sensitivities))

detector_id = 0
for detector in fhc:
    if detector.detector_id not in EXCLUDED_CHANNELS:
        los_radiance_sensitivity = detector._los_radiance_sensitivity.sensitivity
        vol_power_sensitivity = detector._volume_power_sensitivity.sensitivity

        l_los = los_radiance_sensitivity.sum()
        l_vol = vol_power_sensitivity.sum()
        los_to_vol_factor = l_vol / l_los
        los_weight_matrix[detector_id, :] = los_radiance_sensitivity * los_to_vol_factor
        vol_weight_matrix[detector_id, :] = vol_power_sensitivity
        detector_id += 1

        print()
        print(detector.detector_id)
        if detector.detector_id == 'FHC1_A_CH1':
            print('FHC1_A_CH1', True)
        print(l_los, l_vol, los_to_vol_factor)

for detector in flh:
    if detector.detector_id not in EXCLUDED_CHANNELS:
        los_radiance_sensitivity = detector._los_radiance_sensitivity.sensitivity
        vol_power_sensitivity = detector._volume_power_sensitivity.sensitivity

        l_los = los_radiance_sensitivity.sum()
        l_vol = vol_power_sensitivity.sum()
        los_to_vol_factor = l_vol / l_los
        los_weight_matrix[detector_id, :] = los_radiance_sensitivity * los_to_vol_factor
        vol_weight_matrix[detector_id, :] = vol_power_sensitivity
        detector_id += 1

for detector in fhs:
    if detector.detector_id not in EXCLUDED_CHANNELS:
        los_radiance_sensitivity = detector._los_radiance_sensitivity.sensitivity
        vol_power_sensitivity = detector._volume_power_sensitivity.sensitivity

        l_los = los_radiance_sensitivity.sum()
        l_vol = vol_power_sensitivity.sum()
        los_to_vol_factor = l_vol / l_los
        los_weight_matrix[detector_id, :] = los_radiance_sensitivity * los_to_vol_factor
        vol_weight_matrix[detector_id, :] = vol_power_sensitivity
        detector_id += 1

for detector in fvc:
    if detector.detector_id not in EXCLUDED_CHANNELS:
        los_radiance_sensitivity = detector._los_radiance_sensitivity.sensitivity
        vol_power_sensitivity = detector._volume_power_sensitivity.sensitivity

        l_los = los_radiance_sensitivity.sum()
        l_vol = vol_power_sensitivity.sum()
        los_to_vol_factor = l_vol / l_los
        los_weight_matrix[detector_id, :] = los_radiance_sensitivity * los_to_vol_factor
        vol_weight_matrix[detector_id, :] = vol_power_sensitivity
        detector_id += 1

for detector in fdc:
    if detector.detector_id not in EXCLUDED_CHANNELS:
        los_radiance_sensitivity = detector._los_radiance_sensitivity.sensitivity
        vol_power_sensitivity = detector._volume_power_sensitivity.sensitivity

        l_los = los_radiance_sensitivity.sum()
        l_vol = vol_power_sensitivity.sum()
        los_to_vol_factor = l_vol / l_los
        los_weight_matrix[detector_id, :] = los_radiance_sensitivity * los_to_vol_factor
        vol_weight_matrix[detector_id, :] = vol_power_sensitivity
        detector_id += 1

for detector in flx:
    if detector.detector_id not in EXCLUDED_CHANNELS:
        los_radiance_sensitivity = detector._los_radiance_sensitivity.sensitivity
        vol_power_sensitivity = detector._volume_power_sensitivity.sensitivity

        l_los = los_radiance_sensitivity.sum()
        l_vol = vol_power_sensitivity.sum()
        los_to_vol_factor = l_vol / l_los
        los_weight_matrix[detector_id, :] = los_radiance_sensitivity * los_to_vol_factor
        vol_weight_matrix[detector_id, :] = vol_power_sensitivity
        detector_id += 1

print("los_weight_matrix condition", np.linalg.cond(los_weight_matrix))
print("vol_weight_matrix condition", np.linalg.cond(vol_weight_matrix))

print(los_weight_matrix.sum(), vol_weight_matrix.sum())

plt.ion()
los_combined = SensitivityMatrix(grid, "los_combined", sensitivity=los_weight_matrix.sum(axis=0))
los_combined.plot()
plt.axis('equal')

vol_combined = SensitivityMatrix(grid, "vol_combined", sensitivity=vol_weight_matrix.sum(axis=0))
vol_combined.plot()
plt.axis('equal')


