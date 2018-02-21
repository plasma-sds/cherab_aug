
import re
import numpy as np
import pickle
import matplotlib.pyplot as plt

from raysect.optical import World

from cherab.aug.bolometry import FDC_TUBE, FLX_TUBE, FVC_TUBE, FHS_TUBE, load_default_bolometer_config
from cherab.aug.bolometry import load_standard_inversion_grid

# Load default inversion grid
grid = load_standard_inversion_grid()

world = World()
fvc = load_default_bolometer_config('FVC', parent=world)
flx = load_default_bolometer_config('FLX', parent=world)
fdc = load_default_bolometer_config('FDC', parent=world)
fhs = load_default_bolometer_config('FHS', parent=world)
flh = load_default_bolometer_config('FLH', parent=world)
fhc = load_default_bolometer_config('FHC', parent=world)
detectors = fvc.foil_detectors + flx.foil_detectors + fdc.foil_detectors + fhs.foil_detectors + flh.foil_detectors + fhc.foil_detectors

etendue_error_factor = {}


dtype = np.dtype([('camera', '|S3'), ('channel', np.int32), ('f_Blende', np.float32), ('f_Folie', np.float32),
                   ('d_folie_blende', np.float32), ('delta', np.float32), ('gamma', np.float32),
                   ('alpha', np.float32), ('faktor_f', np.float32)])
mb_etendue_array = np.loadtxt('34252.txt', dtype=dtype, skiprows=23, usecols=(0, 1, 7, 8, 9, 10, 11, 12, 19))

fvc_rows = []
flx_rows = []
fdc_rows = []
fhs_rows = []
flh_rows = []
fhc_rows = []
for row in mb_etendue_array:
    camera_id = row[0].decode('UTF-8')
    if camera_id == "FVC":
        fvc_rows.append(row)
    elif camera_id == "FLX":
        flx_rows.append(row)
    elif camera_id == "FDC":
        fdc_rows.append(row)
    elif camera_id == "FHS":
        fhs_rows.append(row)
    elif camera_id == "FLH":
        flh_rows.append(row)
    elif camera_id == "FHC":
        fhc_rows.append(row)


def percentage_etendue_error(rows):

    global detectors, etendue_error_factor

    aug_etendues = []
    cherab_etendues = []

    percentage_error = []
    for row in rows:
        camera_id = row[0].decode('UTF-8')
        channel_id = row[1]

        for detector in detectors:
            match = re.match('^([A-Z]*)[0-9]?_[A-Z]_CH([0-9]*)$', detector.detector_id)
            if match:
                cherab_camera_id = match.group(1)
                cherab_channel_id = int(match.group(2))

                if cherab_camera_id == camera_id and cherab_channel_id == channel_id:
                    cherab_detector = detector
                    break
        else:
            print()
            print("Detector combination camera '{}' and channel '{}' was not found.".format(camera_id, channel_id))
            print()
            continue

        mb_area_pinhole = row[2]  # f_Blende
        mb_area_foil = row[3]  # f_Folie
        mb_d = row[4]  # d_folie_blende
        mb_delta = row[5]  # delta
        mb_gamma = row[6]  # gamma
        mb_alpha = row[7]  # alpha
        mb_f_factor = row[8]
        mb_etendue = mb_f_factor * 4 * np.pi

        mb_mesh_area_reduction = mb_area_pinhole/(detector.slit.dx * detector.slit.dy)
        mb_omega = mb_area_pinhole * np.cos(np.deg2rad(mb_gamma)) / mb_d**2
        mb_calc_etendue = np.cos(np.deg2rad(mb_alpha)) * mb_area_foil * mb_omega

        cherab_detector.calculate_etendue(ray_count=100000)
        cherab_corrected_etendue = cherab_detector.etendue * mb_mesh_area_reduction
        cherab_percent_error = np.abs(mb_etendue-cherab_corrected_etendue)/mb_etendue * 100
        percentage_error.append(cherab_percent_error)

        etendue_factor = mb_etendue/cherab_corrected_etendue
        if 0.7 < etendue_factor < 1.3:
            etendue_error_factor[detector.detector_id] = etendue_factor
        else:
            etendue_error_factor[detector.detector_id] = 1.0

        print('{}'.format(detector.detector_id),
              'MB {:.4G} m^2 str'.format(mb_etendue),
              'CH {:.4G} m^2 str'.format(cherab_corrected_etendue),
              'ER {:.4G} %'.format(cherab_percent_error))
        aug_etendues.append(mb_etendue)
        cherab_etendues.append(cherab_corrected_etendue)

    return percentage_error, aug_etendues, cherab_etendues


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


plt.ion()
percentage_errors, aug_etendues, cherab_etendues = percentage_etendue_error(fvc_rows)
filtered_percentage_error, outliers, nans = remove_outliers(percentage_errors)
plt.figure()
plt.plot(aug_etendues, label='Analytic etendue')
plt.plot(cherab_etendues, label='ray-traced etendue')
plt.title('FVC etendues')
print()
print("FVC performance")
print(np.mean(filtered_percentage_error), np.std(filtered_percentage_error))
print('with n = {} outliers'.format(len(outliers)))
print('with n = {} nans'.format(len(nans)))
print()
print()
percentage_errors, aug_etendues, cherab_etendues = percentage_etendue_error(flx_rows)
filtered_percentage_error, outliers, nans = remove_outliers(percentage_errors)
plt.figure()
plt.plot(aug_etendues, label='Analytic etendue')
plt.plot(cherab_etendues, label='ray-traced etendue')
plt.title('FLX etendues')
print()
print("FLX performance")
print(np.mean(filtered_percentage_error), np.std(filtered_percentage_error))
print('with n = {} outliers'.format(len(outliers)))
print('with n = {} nans'.format(len(nans)))
print()
print()
percentage_errors, aug_etendues, cherab_etendues = percentage_etendue_error(fdc_rows)
filtered_percentage_error, outliers, nans = remove_outliers(percentage_errors)
plt.figure()
plt.plot(aug_etendues, label='Analytic etendue')
plt.plot(cherab_etendues, label='ray-traced etendue')
plt.title('FDC etendues')
print()
print("FDC performance")
print(np.mean(filtered_percentage_error), np.std(filtered_percentage_error))
print('with n = {} outliers'.format(len(outliers)))
print('with n = {} nans'.format(len(nans)))
print()
print()
percentage_errors, aug_etendues, cherab_etendues = percentage_etendue_error(fhs_rows)
filtered_percentage_error, outliers, nans = remove_outliers(percentage_errors)
plt.figure()
plt.plot(aug_etendues, label='Analytic etendue')
plt.plot(cherab_etendues, label='ray-traced etendue')
plt.title('FHS etendues')
print()
print("FHS performance")
print(np.mean(filtered_percentage_error), np.std(filtered_percentage_error))
print('with n = {} outliers'.format(len(outliers)))
print('with n = {} nans'.format(len(nans)))
print()
print()
percentage_errors, aug_etendues, cherab_etendues = percentage_etendue_error(flh_rows)
filtered_percentage_error, outliers, nans = remove_outliers(percentage_errors)
plt.figure()
plt.plot(aug_etendues, label='Analytic etendue')
plt.plot(cherab_etendues, label='ray-traced etendue')
plt.title('FLH etendues')
print()
print("FLH performance")
print(np.mean(filtered_percentage_error), np.std(filtered_percentage_error))
print('with n = {} outliers'.format(len(outliers)))
print('with n = {} nans'.format(len(nans)))
print()
print()
percentage_errors, aug_etendues, cherab_etendues = percentage_etendue_error(fhc_rows)
filtered_percentage_error, outliers, nans = remove_outliers(percentage_errors)
plt.figure()
plt.plot(aug_etendues, label='Analytic etendue')
plt.plot(cherab_etendues, label='ray-traced etendue')
plt.title('FHC etendues')
print()
print("FHC performance")
print(np.mean(filtered_percentage_error), np.std(filtered_percentage_error))
print('with n = {} outliers'.format(len(outliers)))
print('with n = {} nans'.format(len(nans)))


# Save etendue correction factors
fh = open('aug_etendue_error_factor.pickle', 'wb')
pickle.dump(etendue_error_factor, fh)
fh.close()


# print()
# print()
# print("Ray Noise Test")
#
# fhc27 = fhc['FHC2_G_CH27']
#
# etendue = []
# fhc27.calculate_etendue(ray_count=100)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=100)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=100)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=100)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=100)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=100)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=100)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=100)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=100)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=100)
# etendue.append(fhc27.etendue)
# print(np.mean(etendue), np.std(etendue))
#
# etendue = []
# fhc27.calculate_etendue(ray_count=1000)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=1000)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=1000)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=1000)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=1000)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=1000)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=1000)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=1000)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=1000)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=1000)
# etendue.append(fhc27.etendue)
# print(np.mean(etendue), np.std(etendue))
#
# etendue = []
# fhc27.calculate_etendue(ray_count=10000)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=10000)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=10000)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=10000)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=10000)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=10000)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=10000)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=10000)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=10000)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=10000)
# etendue.append(fhc27.etendue)
# print(np.mean(etendue), np.std(etendue))
#
# etendue = []
# fhc27.calculate_etendue(ray_count=100000)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=100000)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=100000)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=100000)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=100000)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=100000)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=100000)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=100000)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=100000)
# etendue.append(fhc27.etendue)
# fhc27.calculate_etendue(ray_count=100000)
# etendue.append(fhc27.etendue)
# print(np.mean(etendue), np.std(etendue))

