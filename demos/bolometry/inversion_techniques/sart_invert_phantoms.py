
import numpy as np
import pickle
import scipy
import matplotlib.pyplot as plt

from cherab.tools.inversions import invert_sart, invert_constrained_sart
from cherab.tools.observers.inversion_grid import EmissivityGrid
from cherab.tools.observers.bolometry import assemble_weight_matrix
from cherab.aug.bolometry import load_emissivity_phantom, load_standard_inversion_grid, load_default_bolometer_config


# 'FVC1_A_CH1', 'FVC2_E_CH17', 'FLX_A_CH1', 'FHS_A_CH1'
EXCLUDED_CHANNELS = ['FDC_A_CH1', 'FDC_G_CH28', 'FHC1_A_CH1', 'FHC1_A_CH2', 'FHC1_A_CH3']

NOISE_VARIANCE = 0.04


emiss77 = load_emissivity_phantom('AUG_emission_phantom_077')
emiss = emiss77.emissivities

grid = load_standard_inversion_grid()

laplace_fh = open("/home/matt/CCFE/cherab/aug/cherab/aug/bolometry/grid_construction/grid_laplacian.ndarray", "rb")
GRID_LAPLACIAN = np.load(laplace_fh)
laplace_fh.close()

fhc = load_default_bolometer_config('FHC', inversion_grid=grid)
flh = load_default_bolometer_config('FLH', inversion_grid=grid)
fhs = load_default_bolometer_config('FHS', inversion_grid=grid)
fvc = load_default_bolometer_config('FVC', inversion_grid=grid)
fdc = load_default_bolometer_config('FDC', inversion_grid=grid)
flx = load_default_bolometer_config('FLX', inversion_grid=grid)


detector_keys, los_weight_matrix, vol_weight_matrix = assemble_weight_matrix([fhc, flh, fhs, fvc, fdc, flx], excluded_detectors=EXCLUDED_CHANNELS)

etendue_error_factor_dict = pickle.load(open('/home/matt/CCFE/cherab/aug/demos/bolometry/etendue_comparison/aug_etendue_error_factor.pickle', 'rb'))
etendue_error_factor = np.zeros(len(detector_keys))
for i in range(len(detector_keys)):
    etendue_error_factor[i] = etendue_error_factor_dict[detector_keys[i]]


# Note - only the volume observed power is valid, since that is what is actually measured.
vol_obs_power = np.dot(vol_weight_matrix, emiss77.emissivities)
vol_obs_with_noise = vol_obs_power * (np.random.randn(len(vol_obs_power)) * NOISE_VARIANCE + 1)
# Note - sightline version is same as obs version but with different etendue, as calculated per approximate formula
# This is being applied as a correction factor, which effectively adds extra noise.
los_obs_power = vol_obs_power / etendue_error_factor
los_obs_with_noise = vol_obs_with_noise / etendue_error_factor  # keep noise the same in both data sets


plt.ion()
emiss77.plot()
plt.axis('equal')
print('Phantom total power - {:.4G}MW'.format(emiss77.total_radiated_power()/1E6))


# inverted_emiss_vector, conv = invert_sart(los_weight_matrix, los_obs_power, max_iterations=100)
# inverted_emiss = EmissivityGrid(grid, case_id='Phantom 77 - LOS SART method', emissivities=inverted_emiss_vector)
# inverted_emiss.plot()
# plt.axis('equal')
# print()
# print('SART LOS Inversion total power - {:.4G}MW'.format(inverted_emiss.total_radiated_power()/1E6))
# print('Total iterations', len(conv), 'convergence', conv[-1])
# print('SART RAW LOS correlation - {:.3G}'.format(np.corrcoef(emiss77.emissivities, inverted_emiss.emissivities)[0][1]))

inverted_emiss_vector, conv = invert_sart(los_weight_matrix, los_obs_with_noise, max_iterations=100)
inverted_emiss = EmissivityGrid(grid, case_id='Phantom 77 - LOS SART with noise', emissivities=inverted_emiss_vector)
inverted_emiss.plot()
plt.axis('equal')
print()
print('SART LOS with noise total power - {:.4G}MW'.format(inverted_emiss.total_radiated_power()/1E6))
print('Total iterations', len(conv), 'convergence', conv[-1])
print('SART RAW + NOISE LOS correlation - {:.3G}'.format(np.corrcoef(emiss77.emissivities, inverted_emiss.emissivities)[0][1]))

# inverted_emiss_vector, conv = invert_constrained_sart(los_weight_matrix, GRID_LAPLACIAN, los_obs_power, max_iterations=300, beta_laplace=80)
# inverted_emiss = EmissivityGrid(grid, case_id='Phantom 77 - LOS Constrained C-SART method', emissivities=np.squeeze(np.asarray(inverted_emiss_vector)))
# inverted_emiss.plot()
# plt.axis('equal')
# print()
# print('Constrained C-SART LOS Inversion total power - {:.4G}MW'.format(inverted_emiss.total_radiated_power()/1E6))
# print('Total iterations', len(conv), 'convergence', conv[-1])
# print('C-SART LOS correlation - {:.3G}'.format(np.corrcoef(emiss77.emissivities, inverted_emiss.emissivities)[0][1]))

inverted_emiss_vector, conv = invert_constrained_sart(los_weight_matrix, GRID_LAPLACIAN, los_obs_with_noise, max_iterations=300, beta_laplace=80)
inverted_emiss = EmissivityGrid(grid, case_id='Phantom 77 - LOS Constrained C-SART with Noise', emissivities=np.squeeze(np.asarray(inverted_emiss_vector)))
inverted_emiss.plot()
plt.axis('equal')
print()
print('Constrained C-SART LOS + Noise total power - {:.4G}MW'.format(inverted_emiss.total_radiated_power()/1E6))
print('Total iterations', len(conv), 'convergence', conv[-1])
print('C-SART LOS + Noise correlation - {:.3G}'.format(np.corrcoef(emiss77.emissivities, inverted_emiss.emissivities)[0][1]))


# inverted_emiss_vector, conv = invert_sart(vol_weight_matrix, vol_obs_power, max_iterations=100)
# inverted_emiss = EmissivityGrid(grid, case_id='Phantom 77 - VOL SART method', emissivities=inverted_emiss_vector)
# inverted_emiss.plot()
# plt.axis('equal')
# print()
# print('SART Volume Inversion total power - {:.4G}MW'.format(inverted_emiss.total_radiated_power()/1E6))
# print('Total iterations', len(conv), 'convergence', conv[-1])
# print('SART RAW Volume correlation - {:.3G}'.format(np.corrcoef(emiss77.emissivities, inverted_emiss.emissivities)[0][1]))

inverted_emiss_vector, conv = invert_sart(vol_weight_matrix, vol_obs_with_noise, max_iterations=100)
inverted_emiss = EmissivityGrid(grid, case_id='Phantom 77 - VOL SART with noise', emissivities=inverted_emiss_vector)
inverted_emiss.plot()
plt.axis('equal')
print()
print('SART Volume + Noise total power - {:.4G}MW'.format(inverted_emiss.total_radiated_power()/1E6))
print('Total iterations', len(conv), 'convergence', conv[-1])
print('SART RAW Volume + Noise correlation - {:.3G}'.format(np.corrcoef(emiss77.emissivities, inverted_emiss.emissivities)[0][1]))

# inverted_emiss_vector, conv = invert_constrained_sart(vol_weight_matrix, GRID_LAPLACIAN, vol_obs_power, max_iterations=300, beta_laplace=80)
# inverted_emiss = EmissivityGrid(grid, case_id='Phantom 77 - VOL Constrained C-SART method', emissivities=np.squeeze(np.asarray(inverted_emiss_vector)))
# inverted_emiss.plot()
# plt.axis('equal')
# print()
# print('Constrained C-SART Volume Inversion total power - {:.4G}MW'.format(inverted_emiss.total_radiated_power()/1E6))
# print('Total iterations', len(conv), 'convergence', conv[-1])
# print('C-SART Volume correlation - {:.3G}'.format(np.corrcoef(emiss77.emissivities, inverted_emiss.emissivities)[0][1]))

inverted_emiss_vector, conv = invert_constrained_sart(vol_weight_matrix, GRID_LAPLACIAN, vol_obs_with_noise, max_iterations=300, beta_laplace=80)
inverted_emiss = EmissivityGrid(grid, case_id='Phantom 77 - VOL C-SART with Noise', emissivities=np.squeeze(np.asarray(inverted_emiss_vector)))
inverted_emiss.plot()
plt.axis('equal')
print()
print('C-SART Volume + Noise total power - {:.4G}MW'.format(inverted_emiss.total_radiated_power()/1E6))
print('Total iterations', len(conv), 'convergence', conv[-1])
print('C-SART Volume + Noise correlation - {:.3G}'.format(np.corrcoef(emiss77.emissivities, inverted_emiss.emissivities)[0][1]))
