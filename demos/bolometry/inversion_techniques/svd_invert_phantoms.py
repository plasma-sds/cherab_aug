
import numpy as np
import scipy
import matplotlib.pyplot as plt

from cherab.tools.inversions import invert_svd
from cherab.tools.observers.inversion_grid import EmissivityGrid
from cherab.tools.observers.bolometry import assemble_weight_matrix
from cherab.aug.bolometry import load_emissivity_phantom, load_standard_inversion_grid, load_default_bolometer_config


EXCLUDED_CHANNELS = ['FVC1_A_CH1', 'FVC2_E_CH17', 'FLX_A_CH1', 'FDC_A_CH1', 'FDC_G_CH28', 'FHS_A_CH1',
                     'FHC1_A_CH1', 'FHC1_A_CH2', 'FHC1_A_CH3']


emiss77 = load_emissivity_phantom('AUG_emission_phantom_077')
emiss = emiss77.emissivities

grid = load_standard_inversion_grid()

fhc = load_default_bolometer_config('FHC', inversion_grid=grid)
flh = load_default_bolometer_config('FLH', inversion_grid=grid)
fhs = load_default_bolometer_config('FHS', inversion_grid=grid)
fvc = load_default_bolometer_config('FVC', inversion_grid=grid)
fdc = load_default_bolometer_config('FDC', inversion_grid=grid)
flx = load_default_bolometer_config('FLX', inversion_grid=grid)


detector_keys, los_weight_matrix, vol_weight_matrix = assemble_weight_matrix([fhc, flh, fhs, fvc, fdc, flx],
                                                                             excluded_detectors=EXCLUDED_CHANNELS)

print("vol_weight_matrix shape", vol_weight_matrix.shape)
fhc13_real = fhc[detector_keys[9]]._volume_power_sensitivity.sensitivity
test_fhc13 = vol_weight_matrix[9, :]

fhc13_obs_power = 0
for i in range(len(test_fhc13)):
    if fhc13_real[i] != test_fhc13[i]:
        print('Warning {} != {}'.format(fhc13_real[i], test_fhc13[i]))

    fhc13_obs_power += fhc13_real[i] * emiss[i]


obs_power = np.dot(vol_weight_matrix, emiss77.emissivities)

print("fhc13_obs_power", fhc13_obs_power)
print("obs_power", obs_power[9])

inverted_emiss_vector = invert_svd(vol_weight_matrix, obs_power)
inverted_emiss = EmissivityGrid(grid, case_id='Phantom 77 - LOS method', emissivities=inverted_emiss_vector)

fhc13_inverted_power = 0
for i in range(len(test_fhc13)):
    fhc13_inverted_power += fhc13_real[i] * inverted_emiss_vector[i]
print("fhc13_inverted_power", fhc13_inverted_power)


plt.ion()
emiss77.plot()
plt.axis('equal')

inverted_emiss.plot()
plt.axis('equal')


print('Phantom total power', emiss77.total_radiated_power())
print('Inversion total power', inverted_emiss.total_radiated_power())

