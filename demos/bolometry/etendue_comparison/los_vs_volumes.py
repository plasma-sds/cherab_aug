
import re
import numpy as np
import matplotlib.pyplot as plt

from raysect.optical import World

from cherab.aug.bolometry import FDC_TUBE, FLX_TUBE, FVC_TUBE, FHS_TUBE, load_default_bolometer_config
from cherab.aug.bolometry import load_standard_inversion_grid

# Load default inversion grid
grid = load_standard_inversion_grid()

world = World()
fvc = load_default_bolometer_config('FVC', parent=world, inversion_grid=grid)
flx = load_default_bolometer_config('FLX', parent=world, inversion_grid=grid)
fdc = load_default_bolometer_config('FDC', parent=world, inversion_grid=grid)
fhs = load_default_bolometer_config('FHS', parent=world, inversion_grid=grid)
flh = load_default_bolometer_config('FLH', parent=world, inversion_grid=grid)
fhc = load_default_bolometer_config('FHC', parent=world, inversion_grid=grid)


def los_vs_volume_power(detectors):

    percentage_errors = []

    for detector in detectors:

        detector.calculate_etendue()

        los_unity_power = detector._los_radiance_sensitivity.sensitivity.sum() * detector.etendue
        vol_unity_power = detector._volume_radiance_sensitivity.sensitivity.sum() * detector._volume_observer.etendue

        percent_error = np.abs(los_unity_power-vol_unity_power)/vol_unity_power * 100
        percentage_errors.append(percent_error)

        print('{}'.format(detector.detector_id),
              'LOS {:.4G} W'.format(los_unity_power),
              'VOL {:.4G} W'.format(vol_unity_power),
              'ER {:.4G} %'.format(percent_error))

    return percentage_errors


# this function from - https://gist.github.com/vishalkuo/f4aec300cf6252ed28d3
def remove_outliers(x, outlier_constant=10):
    nans = []
    valid_data = []
    for y in x:
        if np.isnan(y):
            nans.append(y)
        else:
            valid_data.append(y)
    valid_data = np.array(valid_data)
    upper_quartile = np.percentile(valid_data, 75)
    lower_quartile = np.percentile(valid_data, 25)
    iqr = (upper_quartile - lower_quartile) * outlier_constant
    quartileset = (lower_quartile - iqr, upper_quartile + iqr)
    resultList = []
    outliers = []
    for y in valid_data:
        if quartileset[0] <= y <= quartileset[1]:
            resultList.append(y)
            resultList.append(y)
        else:
            outliers.append(y)
    return resultList, outliers, nans


percentage_errors = los_vs_volume_power(fvc.foil_detectors)
filtered_percentage_error, outliers, nans = remove_outliers(percentage_errors)
print()
print("FVC performance")
print(np.mean(filtered_percentage_error), np.std(filtered_percentage_error))
print('with n = {} outliers'.format(len(outliers)))
print('with n = {} nans'.format(len(nans)))
print()
print()
percentage_errors = los_vs_volume_power(flx.foil_detectors)
filtered_percentage_error, outliers, nans = remove_outliers(percentage_errors)
print()
print("FLX performance")
print(np.mean(filtered_percentage_error), np.std(filtered_percentage_error))
print('with n = {} outliers'.format(len(outliers)))
print('with n = {} nans'.format(len(nans)))
print()
print()
percentage_errors = los_vs_volume_power(fdc.foil_detectors)
filtered_percentage_error, outliers, nans = remove_outliers(percentage_errors)
print()
print("FDC performance")
print(np.mean(filtered_percentage_error), np.std(filtered_percentage_error))
print('with n = {} outliers'.format(len(outliers)))
print('with n = {} nans'.format(len(nans)))
print()
print()
percentage_errors = los_vs_volume_power(fhs.foil_detectors)
filtered_percentage_error, outliers, nans = remove_outliers(percentage_errors)
print()
print("FHS performance")
print(np.mean(filtered_percentage_error), np.std(filtered_percentage_error))
print('with n = {} outliers'.format(len(outliers)))
print('with n = {} nans'.format(len(nans)))
print()
print()
percentage_errors = los_vs_volume_power(flh.foil_detectors)
filtered_percentage_error, outliers, nans = remove_outliers(percentage_errors)
print()
print("FLH performance")
print(np.mean(filtered_percentage_error), np.std(filtered_percentage_error))
print('with n = {} outliers'.format(len(outliers)))
print('with n = {} nans'.format(len(nans)))
print()
print()
percentage_errors = los_vs_volume_power(fhc.foil_detectors)
filtered_percentage_error, outliers, nans = remove_outliers(percentage_errors)
print()
print("FHC performance")
print(np.mean(filtered_percentage_error), np.std(filtered_percentage_error))
print('with n = {} outliers'.format(len(outliers)))
print('with n = {} nans'.format(len(nans)))
