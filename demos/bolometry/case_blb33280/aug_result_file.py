
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from raysect.core import World

from cherab.tools.observers import find_wall_intersection
from cherab.tools.observers.inversion_grid import EmissivityGrid
from cherab.aug.bolometry import load_standard_inversion_grid, load_blb_config_file, load_blb_result_file, load_default_bolometer_config
from cherab.aug.machine import plot_aug_wall_outline, import_mesh_segment, VESSEL, PSL, ICRH, DIVERTOR, A_B_COILS
from cherab.aug.bolometry import FDC_TUBE, FLX_TUBE, FVC_TUBE, FHS_TUBE, load_default_bolometer_config


# Make Latex available in matplotlib figures
rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})


SCALE_CORRECTION = 1000/365

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

etendue_error_factor_dict = pickle.load(open('/home/matt/CCFE/cherab/aug/demos/bolometry/etendue_comparison/aug_etendue_error_factor.pickle', 'rb'))


def process_detector_error(detector, emission_profile):

    los_sensitivity = detector._los_radiance_sensitivity.sensitivity
    vol_sensitivity = detector._volume_radiance_sensitivity.sensitivity
    vol_etendue = detector._volume_observer.etendue

    l_los = los_sensitivity.sum()
    l_vol = vol_sensitivity.sum()
    los_to_vol_factor = l_vol / l_los

    los_etendue_error_factor = etendue_error_factor_dict[detector.detector_id]  # error due to approximate etendue
    p_los_geom = np.dot(emission_profile.emissivities, los_sensitivity) * los_to_vol_factor * vol_etendue
    p_los_full = p_los_geom / los_etendue_error_factor
    p_vol_observed = np.dot(emission_profile.emissivities, vol_sensitivity) * vol_etendue
    percent_error_geom = np.abs(p_los_geom - p_vol_observed) / p_vol_observed * 100
    percent_error_full = np.abs(p_los_full - p_vol_observed) / p_vol_observed * 100

    # print()
    # print(detector.detector_id)
    # print("l_vol", l_vol)
    # print("l_los", l_los)
    # print("los_to_vol_factor", los_to_vol_factor)
    # print("p_los_observed", p_los_observed)
    # print("p_vol_observed", p_vol_observed)
    # print("% error", percent_error)

    return p_los_geom, p_los_full, p_vol_observed, percent_error_geom, percent_error_full


def plot_detector_sightline(world, detector, percent_error):

    centre_point = detector.centre_point
    sightline_vec = centre_point.vector_to(detector._slit.centre_point)

    hit_point, primitive = find_wall_intersection(world, centre_point, sightline_vec)

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


geom_model_percent_errors = {}
full_model_percent_errors = {}
flx_los_geom = []
flx_los_full = []
flx_vol_obs = []
for detector in flx:
    if detector.detector_id not in EXCLUDED_CHANNELS:
        p_los_geom, p_los_full, p_vol_observed, percent_error_geom, percent_error_full = process_detector_error(detector, result_blb33280)
        flx_los_geom.append(p_los_geom)
        flx_los_full.append(p_los_full)
        flx_vol_obs.append(p_vol_observed)
        geom_model_percent_errors[percent_error_geom] = (detector, flx_world)
        full_model_percent_errors[percent_error_full] = (detector, flx_world)

fdc_los_geom = []
fdc_los_full = []
fdc_vol_obs = []
for detector in fdc:
    if detector.detector_id not in EXCLUDED_CHANNELS:
        p_los_geom, p_los_full, p_vol_observed, percent_error_geom, percent_error_full = process_detector_error(detector, result_blb33280)
        fdc_los_geom.append(p_los_geom)
        fdc_los_full.append(p_los_full)
        fdc_vol_obs.append(p_vol_observed)
        geom_model_percent_errors[percent_error_geom] = (detector, fdc_world)
        full_model_percent_errors[percent_error_full] = (detector, fdc_world)

fvc_los_geom = []
fvc_los_full = []
fvc_vol_obs = []
for detector in fvc:
    if detector.detector_id not in EXCLUDED_CHANNELS:
        p_los_geom, p_los_full, p_vol_observed, percent_error_geom, percent_error_full = process_detector_error(detector, result_blb33280)
        fvc_los_geom.append(p_los_geom)
        fvc_los_full.append(p_los_full)
        fvc_vol_obs.append(p_vol_observed)
        geom_model_percent_errors[percent_error_geom] = (detector, fvc_world)
        full_model_percent_errors[percent_error_full] = (detector, fvc_world)

fhs_los_geom = []
fhs_los_full = []
fhs_vol_obs = []
for detector in fhs:
    if detector.detector_id not in EXCLUDED_CHANNELS:
        p_los_geom, p_los_full, p_vol_observed, percent_error_geom, percent_error_full = process_detector_error(detector, result_blb33280)
        fhs_los_geom.append(p_los_geom)
        fhs_los_full.append(p_los_full)
        fhs_vol_obs.append(p_vol_observed)
        geom_model_percent_errors[percent_error_geom] = (detector, fhs_world)
        full_model_percent_errors[percent_error_full] = (detector, fhs_world)

fhc_los_geom = []
fhc_los_full = []
fhc_vol_obs = []
for detector in fhc:
    if detector.detector_id not in EXCLUDED_CHANNELS:
        p_los_geom, p_los_full, p_vol_observed, percent_error_geom, percent_error_full = process_detector_error(detector, result_blb33280)
        fhc_los_geom.append(p_los_geom)
        fhc_los_full.append(p_los_full)
        fhc_vol_obs.append(p_vol_observed)
        geom_model_percent_errors[percent_error_geom] = (detector, full_world)
        full_model_percent_errors[percent_error_full] = (detector, full_world)

flh_los_geom = []
flh_los_full = []
flh_vol_obs = []
for detector in flh:
    if detector.detector_id not in EXCLUDED_CHANNELS:
        p_los_geom, p_los_full, p_vol_observed, percent_error_geom, percent_error_full = process_detector_error(detector, result_blb33280)
        flh_los_geom.append(p_los_geom)
        flh_los_full.append(p_los_full)
        flh_vol_obs.append(p_vol_observed)
        geom_model_percent_errors[percent_error_geom] = (detector, full_world)
        full_model_percent_errors[percent_error_full] = (detector, full_world)


plt.ion()
patches = []
for i in range(result_blb33280.count):
    polygon = Polygon(result_blb33280.grid_geometry.cell_data[i], True)
    patches.append(polygon)

p = PatchCollection(patches)
p.set_array(result_blb33280.emissivities * np.pi * 4 / 1E6)

fig, ax = plt.subplots()
ax.add_collection(p)
plt.xlim(1, 2.5)
plt.ylim(-1.5, 1.5)
title = result_blb33280.case_id + " - Emissivity"
plt.title(title)
plt.axis('equal')
fig.colorbar(p, ax=ax)
plot_aug_wall_outline()


straight_line = np.linspace(0, 100)

# plt.figure()
# plt.plot(straight_line, straight_line, 'k--')
# plt.plot(fhc_los_geom, fhc_vol_obs, 'b.')
# plt.plot(fdc_los_geom, fdc_vol_obs, 'b.')
# plt.plot(flx_los_geom, flx_vol_obs, 'b.')
# plt.plot(fvc_los_geom, fvc_vol_obs, 'b.')
# plt.plot(fhs_los_geom, fhs_vol_obs, 'b.')
# plt.plot(flh_los_geom, flh_vol_obs, 'b.')
# plt.xlabel(r"$\Phi_{SR}$ - Power observed with single-ray method (W)")
# plt.ylabel(r"$\Phi_{Vol}$ - Power observed with ray-tracing method (W)")
# plt.title("Observed power with single-ray VS ray-tracing - 33280 4.1s")
# plt.xlim(0, 8)
# plt.ylim(0, 8)
# plt.legend()

plt.figure()
plt.plot(straight_line, straight_line, 'k--')
plt.plot(straight_line, straight_line*0.9, color="grey", linestyle='--')
plt.plot(straight_line, straight_line*1.1, color="grey", linestyle='--')
plt.plot(np.array(fhc_los_full)*SCALE_CORRECTION, np.array(fhc_vol_obs)*SCALE_CORRECTION, 'b.')
plt.plot(np.array(fdc_los_full)*SCALE_CORRECTION, np.array(fdc_vol_obs)*SCALE_CORRECTION, 'b.')
plt.plot(np.array(flx_los_full)*SCALE_CORRECTION, np.array(flx_vol_obs)*SCALE_CORRECTION, 'b.')
plt.plot(np.array(fvc_los_full)*SCALE_CORRECTION, np.array(fvc_vol_obs)*SCALE_CORRECTION, 'b.')
plt.plot(np.array(fhs_los_full)*SCALE_CORRECTION, np.array(fhs_vol_obs)*SCALE_CORRECTION, 'b.')
plt.plot(np.array(flh_los_full)*SCALE_CORRECTION, np.array(flh_vol_obs)*SCALE_CORRECTION, 'b.')
plt.xlabel(r"$\Phi_{SR}$ - Power observed with single-ray method (mW)")
plt.ylabel(r"$\Phi_{Vol}$ - Power observed with ray-tracing method (mW)")
plt.title("Observed power with single-ray VS ray-tracing - 33280 4.1s")
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.legend()


plt.figure()
plt.plot(straight_line, straight_line, 'k--')
plt.plot(straight_line, straight_line*0.9, color="grey", linestyle='--')
plt.plot(straight_line, straight_line*1.1, color="grey", linestyle='--')
plt.plot(np.array(fhc_los_full)*SCALE_CORRECTION, np.array(fhc_vol_obs)*SCALE_CORRECTION, 'b.')
plt.plot(np.array(fdc_los_full)*SCALE_CORRECTION, np.array(fdc_vol_obs)*SCALE_CORRECTION, 'b.')
plt.plot(np.array(flx_los_full)*SCALE_CORRECTION, np.array(flx_vol_obs)*SCALE_CORRECTION, 'b.')
plt.plot(np.array(fvc_los_full)*SCALE_CORRECTION, np.array(fvc_vol_obs)*SCALE_CORRECTION, 'b.')
plt.plot(np.array(fhs_los_full)*SCALE_CORRECTION, np.array(fhs_vol_obs)*SCALE_CORRECTION, 'b.')
plt.plot(np.array(flh_los_full)*SCALE_CORRECTION, np.array(flh_vol_obs)*SCALE_CORRECTION, 'b.')
plt.xlabel(r"$\Phi_{SR}$ - Power observed with single-ray method (mW)")
plt.ylabel(r"$\Phi_{Vol}$ - Power observed with ray-tracing method (mW)")
plt.title("Observed power with single-ray VS ray-tracing - 33280 4.1s")
plt.legend()



plt.figure()
for percent_error in sorted(geom_model_percent_errors.keys()):
    detector, world = geom_model_percent_errors[percent_error]
    plot_detector_sightline(world, detector, percent_error)
plot_aug_wall_outline()
max_error = max(geom_model_percent_errors.keys())
min_error = min(geom_model_percent_errors.keys())
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_error, vmax=max_error))
sm._A = []  # fake up the array of the scalar mappable. Urgh...
plt.colorbar(sm)

# plt.figure()
# for percent_error in sorted(full_model_percent_errors.keys()):
#     detector, world = full_model_percent_errors[percent_error]
#     plot_detector_sightline(world, detector, percent_error)
# plot_aug_wall_outline()
