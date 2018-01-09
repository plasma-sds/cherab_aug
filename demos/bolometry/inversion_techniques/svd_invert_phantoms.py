
import numpy as np
import scipy
import matplotlib.pyplot as plt

from cherab.tools.inversions import invert_svd
from cherab.tools.observers.inversion_grid import EmissivityGrid
from cherab.tools.observers.bolometry import assemble_weight_matrix
from cherab.aug.bolometry import load_emissivity_phantom, load_standard_inversion_grid, load_default_bolometer_config


# 'FVC1_A_CH1', 'FVC2_E_CH17', 'FLX_A_CH1', 'FHS_A_CH1'
EXCLUDED_CHANNELS = ['FDC_A_CH1', 'FDC_G_CH28', 'FHC1_A_CH1', 'FHC1_A_CH2', 'FHC1_A_CH3']


phantom_emiss = load_emissivity_phantom('AUG_emission_phantom_077')
emiss = phantom_emiss.emissivities

grid = load_standard_inversion_grid()

fhc = load_default_bolometer_config('FHC', inversion_grid=grid)
flh = load_default_bolometer_config('FLH', inversion_grid=grid)
fhs = load_default_bolometer_config('FHS', inversion_grid=grid)
fvc = load_default_bolometer_config('FVC', inversion_grid=grid)
fdc = load_default_bolometer_config('FDC', inversion_grid=grid)
flx = load_default_bolometer_config('FLX', inversion_grid=grid)


detector_keys, los_weight_matrix, vol_weight_matrix = assemble_weight_matrix([fhc, flh, fhs, fvc, fdc, flx],
                                                                             excluded_detectors=EXCLUDED_CHANNELS)

obs_power = np.dot(vol_weight_matrix, phantom_emiss.emissivities)

plt.ion()
phantom_emiss.plot()
plt.axis('equal')
print('Phantom total power - {:.4G}MW'.format(phantom_emiss.total_radiated_power()/1E6))


inverted_emiss_vector = invert_svd(los_weight_matrix, obs_power)
inverted_emiss = EmissivityGrid(grid, case_id='Phantom 77 - LOS method', emissivities=inverted_emiss_vector)
inverted_emiss.plot()
plt.axis('equal')
print('SVD LOS Inversion total power - {:.4G}MW'.format(inverted_emiss.total_radiated_power()/1E6))
print('SVD LOS correlation - {:.3G}'.format(np.corrcoef(phantom_emiss.emissivities, inverted_emiss.emissivities)[0][1]))


inverted_emiss_vector = invert_svd(vol_weight_matrix, obs_power)
inverted_emiss = EmissivityGrid(grid, case_id='Phantom 77 - Volume method', emissivities=inverted_emiss_vector)
inverted_emiss.plot()
plt.axis('equal')
print('SVD VOLUME Inversion total power - {:.4G}MW'.format(inverted_emiss.total_radiated_power()/1E6))
print('SVD VOLUME correlation - {:.3G}'.format(np.corrcoef(phantom_emiss.emissivities, inverted_emiss.emissivities)[0][1]))
