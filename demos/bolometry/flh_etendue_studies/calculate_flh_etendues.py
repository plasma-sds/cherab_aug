
import re
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 12})

from raysect.core import Ray as CoreRay
from raysect.primitive import import_stl
from raysect.optical import World
from raysect.optical.material import AbsorbingSurface
from raysect.optical.material.material import NullMaterial
from raysect.core.math.sampler import TargettedHemisphereSampler

from cherab.aug.bolometry import FDC_TUBE, FLX_TUBE, FVC_TUBE, FHS_TUBE, load_default_bolometer_config
from cherab.aug.bolometry import load_standard_inversion_grid

# Load default inversion grid
grid = load_standard_inversion_grid()

world = World()
flh = load_default_bolometer_config('FLH', parent=world)
detectors = flh.foil_detectors

etendue_error_factor = {}


dtype = np.dtype([('camera', '|S3'), ('channel', np.int32), ('f_Blende', np.float32), ('f_Folie', np.float32),
                   ('d_folie_blende', np.float32), ('delta', np.float32), ('gamma', np.float32),
                   ('alpha', np.float32), ('faktor_f', np.float32)])
mb_etendue_array = np.loadtxt('34252.txt', dtype=dtype, skiprows=23, usecols=(0, 1, 7, 8, 9, 10, 11, 12, 19))

flh_rows = []
for row in mb_etendue_array:
    camera_id = row[0].decode('UTF-8')
    if camera_id == "FLH":
        flh_rows.append(row)


def percentage_etendue_error(rows):

    global detectors, etendue_error_factor

    aug_etendues = []
    cherab_etendues = []
    cherab_etendue_errors = []
    percentage_error = []
    area_reduction = []

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
        cherab_etendue_errors.append(cherab_detector.etendue_error * mb_mesh_area_reduction)
        area_reduction.append(mb_mesh_area_reduction)

    return percentage_error, aug_etendues, cherab_etendues, cherab_etendue_errors, area_reduction


def calculate_etendue(detector, world, ray_count=10000, batches=10, print_results=False):

    target = detector.slit.primitive

    detector_transform = detector.to_root()

    # generate bounding sphere and convert to local coordinate system
    sphere = target.bounding_sphere()
    spheres = [(sphere.centre.transform(detector.to_local()), sphere.radius, 1.0)]

    # instance targetted pixel sampler
    targetted_sampler = TargettedHemisphereSampler(spheres)

    etendues = []
    for i in range(batches):

        # sample pixel origins
        origins = detector._volume_observer._point_sampler(samples=ray_count)

        passed = 0.0
        for origin in origins:

            # obtain targetted vector sample
            direction, pdf = targetted_sampler(origin, pdf=True)
            path_weight = (1/ (2 * np.pi)) * direction.z/pdf

            origin = origin.transform(detector_transform)
            direction = direction.transform(detector_transform)

            while True:

                # Find the next intersection point of the ray with the world
                intersection = world.hit(CoreRay(origin, direction))

                if intersection is None:
                    passed += 1 * path_weight
                    break

                elif isinstance(intersection.primitive.material, NullMaterial):
                    hit_point = intersection.hit_point.transform(intersection.primitive_to_world)
                    origin = hit_point + direction * 1E-9
                    continue

                else:
                    break

        if passed == 0:
            raise ValueError("Something is wrong with the scene-graph, calculated etendue should not zero.")

        etendue_fraction = passed / ray_count

        etendues.append(detector._volume_observer.etendue * etendue_fraction)

    etendue = np.mean(etendues)
    etendue_error = np.std(etendues)

    return etendue, etendue_error


plt.ion()
percentage_errors, aug_etendues, cherab_etendues, cherab_etendue_errors, area_reductions = percentage_etendue_error(flh_rows)


# remove the CSG slit from the scene since we are using CAD instead
flh2 = flh['FLH_A_CH2']
flh2.slit.csg_aperture.parent = None

# load detector box
flh_box = import_stl("/home/matt/CCFE/cadmesh/aug/diagnostics/bolometry/FLH_box_no_rfgrid.stl", parent=world,
                     material=AbsorbingSurface(), name="FLH_box", scaling=0.001)

apperature_3d_etendues = []
apperature_3d_etendue_errors = []
for i, detector in enumerate(detectors):
    etendue, etendue_error = calculate_etendue(detector, world)
    apperature_3d_etendues.append(etendue * area_reductions[i])
    apperature_3d_etendue_errors.append(etendue_error * area_reductions[i])


plt.figure(figsize=(8, 4))
plt.plot(np.arange(1, 5), aug_etendues, linestyle='-', marker='o', label='Analytic etendue')
plt.errorbar(np.arange(1, 5), cherab_etendues, yerr=cherab_etendue_errors, label="Simplified pinhole model")
plt.errorbar(np.arange(1, 5), apperature_3d_etendues, yerr=apperature_3d_etendue_errors, label="CAD aperture model")
plt.xticks(np.arange(1, 5), ('#1', '#2', '#3', '#4'))
plt.ylim(0.5E-8, 1.1E-8)
plt.xlabel('Detector ID')
plt.ylabel('Etendue (m^2 str)')
plt.title('Comparison of etendue calculation methods')
plt.legend()

# print()
# print("FLH performance")
# print(np.mean(filtered_percentage_error), np.std(filtered_percentage_error))
# print('with n = {} outliers'.format(len(outliers)))
# print('with n = {} nans'.format(len(nans)))
# print()
# print()



