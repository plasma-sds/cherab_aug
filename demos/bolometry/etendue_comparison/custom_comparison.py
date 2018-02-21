
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


def detailed_etendue_comparison(cherab_detector, ids, rows):

    cherab_camera_id, cherab_channel_id = ids

    for row in rows:
        camera_id = row[0].decode('UTF-8')
        channel_id = row[1]
        # print(camera_id, channel_id, type(channel_id))

        if cherab_camera_id == camera_id and cherab_channel_id == channel_id:

            mb_area_pinhole = row[2]  # f_Blende
            mb_area_foil = row[3]  # f_Folie
            mb_d = row[4]  # d_folie_blende
            mb_delta = row[5]  # delta
            mb_gamma = row[6]  # gamma
            mb_alpha = row[7]  # alpha
            mb_f_factor = row[8]
            mb_etendue = mb_f_factor * 4 * np.pi

            mb_mesh_area_reduction = mb_area_pinhole/(cherab_detector.slit.dx * cherab_detector.slit.dy)
            mb_omega = mb_area_pinhole * np.cos(np.deg2rad(mb_gamma)) / mb_d**2
            mb_calc_etendue = np.cos(np.deg2rad(mb_alpha)) * mb_area_foil * mb_omega

            cherab_detector.calculate_etendue(ray_count=100000)
            cherab_corrected_etendue = cherab_detector.etendue * mb_mesh_area_reduction
            cherab_percent_error = np.abs(mb_etendue-cherab_corrected_etendue)/mb_etendue * 100

            cherab_solid_angle = cherab_detector.etendue / cherab_detector._volume_observer.etendue * 2*np.pi

            print()
            print('MB foil area            {:.4G} m^2'.format(mb_area_foil))
            print('CHERAB foil area        {:.4G} m^2'.format(cherab_detector.dx * cherab_detector.dy))

            print()
            print('MB pinhole area         {:.4G} m^2'.format(mb_area_pinhole))
            print('CHERAB pinhole area     {:.4G} m^2'.format(cherab_detector.slit.dx * cherab_detector.slit.dy))
            print('Area reduction A_frac   {:.4G}'.format(mb_mesh_area_reduction))

            print()
            print('MB p-a distance         {:.4G} m'.format(mb_d))
            print('CHERAB p-a distance     {:.4G} m'.format(cherab_detector.centre_point.distance_to(cherab_detector.slit.centre_point)))

            print()
            print('MB solid angle          {:.4G} str'.format(mb_omega))
            print('CHERAB solid angle      {:.4G} str'.format(cherab_solid_angle))
            print('MB solid angle frac     {:.4G}'.format(mb_omega / (2 * np.pi)))
            print('CHERAB solid angle frac {:.4G}'.format(cherab_detector.etendue / cherab_detector._volume_observer.etendue))

            print()
            print('MB Etendue              {:.4G} m^2 str'.format(mb_etendue * 4 * np.pi))
            print('Formula with MB numbers {:.4G} m^2 str'.format(mb_calc_etendue))
            print('CHERAB etendue          {:.4G} m^2 str'.format(cherab_detector.etendue))
            print('CHERAB etendue * A_frac {:.4G} m^2 str'.format(cherab_detector.etendue * mb_mesh_area_reduction))

            break

    else:
        raise ValueError("Detector combination was not found.")


detailed_etendue_comparison(fvc['FVC1_A_CH1'], ('FVC', 1), fvc_rows)

print()
print('#######################################################################')
print()
detailed_etendue_comparison(fvc['FVC1_A_CH2'], ('FVC', 2), fvc_rows)

