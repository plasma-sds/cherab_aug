
import os
import numpy as np
import matplotlib.pyplot as plt

from raysect.core import World

from cherab.tools.observers import find_wall_intersection
from cherab.tools.observers.inversion_grid import EmissivityGrid
from cherab.aug.bolometry import load_standard_inversion_grid, load_blb_config_file, load_blb_result_file, load_default_bolometer_config
from cherab.aug.machine import plot_aug_wall_outline, import_mesh_segment, VESSEL, PSL, ICRH, DIVERTOR, A_B_COILS
from cherab.aug.bolometry import FDC_TUBE, FLX_TUBE, FVC_TUBE, FHS_TUBE, load_default_bolometer_config


cmap = plt.cm.viridis

# replace with path to your BLB geometry directory
AUG_BLB_CASE_DIRECTORY = os.path.expanduser("~/CCFE/mst1/aug_bolometry/BLB.33280_2")

EXCLUDED_CHANNELS = ['FVC1_A_CH1', 'FVC2_E_CH17', 'FLX_A_CH1', 'FDC_A_CH1', 'FDC_G_CH28', 'FHS_A_CH1'
                     'FHC1_A_CH1', 'FHC1_A_CH2', 'FHC1_A_CH3']

# Config file describing which bolometer cameras are active
grid = load_standard_inversion_grid()

active_cameras = load_blb_config_file(os.path.join(AUG_BLB_CASE_DIRECTORY, "config"))

result_blb33280_raw = load_blb_result_file(os.path.join(AUG_BLB_CASE_DIRECTORY, "resultef2d.BLB.33280_4.100000"), grid)

result_blb33280 = EmissivityGrid(grid, result_blb33280_raw.case_id, result_blb33280_raw.description, emissivities=np.clip(result_blb33280_raw.emissivities, 0, None))


def process_detector_error(detector, emission_profile):

    los_sensitivity = detector._los_radiance_sensitivity.sensitivity
    vol_sensitivity = detector._volume_radiance_sensitivity.sensitivity
    vol_etendue = detector._volume_observer.etendue

    l_los = los_sensitivity.sum()
    l_vol = vol_sensitivity.sum()
    los_to_vol_factor = l_vol / l_los

    p_los_observed = np.dot(emission_profile.emissivities, los_sensitivity) * los_to_vol_factor * vol_etendue
    p_vol_observed = np.dot(emission_profile.emissivities, vol_sensitivity) * vol_etendue
    percent_error = np.abs(p_los_observed - p_vol_observed) / p_vol_observed * 100

    # print()
    # print(detector.detector_id)
    # print("l_vol", l_vol)
    # print("l_los", l_los)
    # print("los_to_vol_factor", los_to_vol_factor)
    # print("p_los_observed", p_los_observed)
    # print("p_vol_observed", p_vol_observed)
    # print("% error", percent_error)

    return p_los_observed, p_vol_observed, percent_error


def plot_detector_sightline(world, detector, percent_error):

    centre_point = detector.centre_point
    sightline_vec = centre_point.vector_to(detector._slit.centre_point)

    hit_point = find_wall_intersection(world, centre_point, sightline_vec)

    # Traverse the ray with equation for a parametric line,
    # i.e. t=0->1 traverses the ray path.
    parametric_vector = centre_point.vector_to(hit_point)
    t_samples = np.arange(0, 1, 0.01)

    # Setup some containers for useful parameters along the ray trajectory
    ray_r_points = []
    ray_z_points = []

    # At each ray position sample the parameters of interest.
    for i, t in enumerate(t_samples):
        # Get new sample point location and log distance
        x = centre_point.x + parametric_vector.x * t
        y = centre_point.y + parametric_vector.y * t
        z = centre_point.z + parametric_vector.z * t
        ray_r_points.append(np.sqrt(x**2 + y**2))
        ray_z_points.append(z)

    plt.plot(ray_r_points, ray_z_points, color=cmap(percent_error/25))
    # plt.plot(ray_r_points[0], ray_z_points[0], 'b.')
    # plt.plot(ray_r_points[-1], ray_z_points[-1], 'r.')


full_world = World()
FULL_MESH_SEGMENTS = VESSEL + PSL + ICRH + DIVERTOR + A_B_COILS
import_mesh_segment(full_world, FULL_MESH_SEGMENTS)
fhc = load_default_bolometer_config('FHC', parent=full_world, inversion_grid=grid)
flh = load_default_bolometer_config('FLH', parent=full_world, inversion_grid=grid)

fhs_world = World()
import_mesh_segment(fhs_world, FHS_TUBE)
fhs = load_default_bolometer_config('FHS', parent=fhs_world, inversion_grid=grid)

fvc_world = World()
import_mesh_segment(fvc_world, FVC_TUBE)
fvc = load_default_bolometer_config('FVC', parent=fvc_world, inversion_grid=grid)

fdc_world = World()
import_mesh_segment(fdc_world, FDC_TUBE)
fdc = load_default_bolometer_config('FDC', parent=fdc_world, inversion_grid=grid)

flx_world = World()
import_mesh_segment(flx_world, FLX_TUBE)
flx = load_default_bolometer_config('FLX', parent=flx_world, inversion_grid=grid)


# Note - FLX_A_CH2 could be a very interesting case, due to way sightline clips the baffle
process_detector_error(flx['FLX_A_CH2'], result_blb33280)
process_detector_error(fhc['FHC3_K_CH41'], result_blb33280)


model_percent_errors = {}

flx_los_obs = []
flx_vol_obs = []
for detector in flx:
    if detector.detector_id not in EXCLUDED_CHANNELS:
        p_los_observed, p_vol_observed, percent_error = process_detector_error(detector, result_blb33280)
        flx_los_obs.append(p_los_observed)
        flx_vol_obs.append(p_vol_observed)
        model_percent_errors[percent_error] = (detector, flx_world)

fdc_los_obs = []
fdc_vol_obs = []
for detector in fdc:
    if detector.detector_id not in EXCLUDED_CHANNELS:
        p_los_observed, p_vol_observed, percent_error = process_detector_error(detector, result_blb33280)
        fdc_los_obs.append(p_los_observed)
        fdc_vol_obs.append(p_vol_observed)
        model_percent_errors[percent_error] = (detector, fdc_world)

fvc_los_obs = []
fvc_vol_obs = []
for detector in fvc:
    if detector.detector_id not in EXCLUDED_CHANNELS:
        p_los_observed, p_vol_observed, percent_error = process_detector_error(detector, result_blb33280)
        fvc_los_obs.append(p_los_observed)
        fvc_vol_obs.append(p_vol_observed)
        model_percent_errors[percent_error] = (detector, fvc_world)

fhs_los_obs = []
fhs_vol_obs = []
for detector in fhs:
    if detector.detector_id not in EXCLUDED_CHANNELS:
        p_los_observed, p_vol_observed, percent_error = process_detector_error(detector, result_blb33280)
        fhs_los_obs.append(p_los_observed)
        fhs_vol_obs.append(p_vol_observed)
        model_percent_errors[percent_error] = (detector, fhs_world)

fhc_los_obs = []
fhc_vol_obs = []
for detector in fhc:
    if detector.detector_id not in EXCLUDED_CHANNELS:
        p_los_observed, p_vol_observed, percent_error = process_detector_error(detector, result_blb33280)
        fhc_los_obs.append(p_los_observed)
        fhc_vol_obs.append(p_vol_observed)
        model_percent_errors[percent_error] = (detector, full_world)

flh_los_obs = []
flh_vol_obs = []
for detector in flh:
    if detector.detector_id not in EXCLUDED_CHANNELS:
        p_los_observed, p_vol_observed, percent_error = process_detector_error(detector, result_blb33280)
        flh_los_obs.append(p_los_observed)
        flh_vol_obs.append(p_vol_observed)
        model_percent_errors[percent_error] = (detector, full_world)


plt.ion()
plt.figure()
result_blb33280.plot()
plot_aug_wall_outline()


straight_line = np.linspace(0, 100)

plt.figure()
plt.plot(straight_line, straight_line, 'k--')
plt.plot(fhc_los_obs, fhc_vol_obs, 'b.')
plt.plot(fdc_los_obs, fdc_vol_obs, 'b.')
plt.plot(flx_los_obs, flx_vol_obs, 'b.')
plt.plot(fvc_los_obs, fvc_vol_obs, 'b.')
plt.plot(fhs_los_obs, fhs_vol_obs, 'b.')
plt.plot(flh_los_obs, flh_vol_obs, 'b.')
plt.xlabel("Power observed with LOS method (W)")
plt.ylabel("Power observed with ray-tracing method (W)")
plt.title("Observed power with LOS VS Ray-tracing - 33280 4.1s")
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.legend()


plt.figure()
for percent_error in sorted(model_percent_errors.keys()):
    detector, world = model_percent_errors[percent_error]
    plot_detector_sightline(world, detector, percent_error)
plot_aug_wall_outline()
