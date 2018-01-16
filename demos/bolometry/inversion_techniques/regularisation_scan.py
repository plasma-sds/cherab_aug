
import numpy as np
import scipy
import matplotlib.pyplot as plt

from cherab.tools.inversions import invert_sart, invert_constrained_sart
from cherab.tools.observers.inversion_grid import EmissivityGrid
from cherab.tools.observers.bolometry import assemble_weight_matrix
from cherab.aug.bolometry import load_emissivity_phantom, load_standard_inversion_grid, load_default_bolometer_config


# 'FVC1_A_CH1', 'FVC2_E_CH17', 'FLX_A_CH1', 'FHS_A_CH1'
EXCLUDED_CHANNELS = ['FDC_A_CH1', 'FDC_G_CH28', 'FHC1_A_CH1', 'FHC1_A_CH2', 'FHC1_A_CH3']


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

cell_ray_densities = np.sum(los_weight_matrix, axis=0)
num_empty_cells = 0
for cell in cell_ray_densities:
    if cell == 0.0:
        num_empty_cells += 1

print(los_weight_matrix.shape)
print("num_empty_cells", num_empty_cells, "empty cell fraction", num_empty_cells / len(cell_ray_densities))

# Note - only the volume observed power is valid, since that is what is actually measured.
obs_power = np.dot(vol_weight_matrix, emiss77.emissivities)


# plt.ion()
# emiss77.plot()
# plt.axis('equal')
# print('Phantom total power - {:.4G}MW'.format(emiss77.total_radiated_power()/1E6))

betas = np.array([10**x for x in np.linspace(1, 3.7, 40)])
los_rhos = []
vol_rhos = []

for beta in betas:

    inverted_emiss_vector, conv = invert_constrained_sart(los_weight_matrix, GRID_LAPLACIAN, obs_power, max_iterations=300, beta_laplace=beta)
    pearson_rho = np.corrcoef(emiss77.emissivities, np.squeeze(np.asarray(inverted_emiss_vector)))[0][1]
    inverted_emiss = EmissivityGrid(grid, case_id='LOS C-SART method - {:.4G}, {:.4G}'.format(beta, pearson_rho), emissivities=np.squeeze(np.asarray(inverted_emiss_vector)))
    # inverted_emiss.plot()
    plt.axis('equal')
    print()
    print('Constrained C-SART LOS Inversion total power - {:.4G}MW'.format(inverted_emiss.total_radiated_power()/1E6))
    print('Total iterations', len(conv), 'convergence', conv[-1])
    print('C-SART LOS correlation - {:.3G}'.format(pearson_rho))
    los_rhos.append(pearson_rho)


    inverted_emiss_vector, conv = invert_constrained_sart(vol_weight_matrix, GRID_LAPLACIAN, obs_power, max_iterations=300, beta_laplace=beta)
    pearson_rho = np.corrcoef(emiss77.emissivities, np.squeeze(np.asarray(inverted_emiss_vector)))[0][1]
    inverted_emiss = EmissivityGrid(grid, case_id='VOL C-SART method', emissivities=np.squeeze(np.asarray(inverted_emiss_vector)))
    # inverted_emiss.plot()
    # plt.axis('equal')
    print()
    print('Constrained C-SART Volume Inversion total power - {:.4G}MW'.format(inverted_emiss.total_radiated_power()/1E6))
    print('Total iterations', len(conv), 'convergence', conv[-1])
    print('C-SART Volume correlation - {:.3G}'.format(np.corrcoef(emiss77.emissivities, inverted_emiss.emissivities)[0][1]))
    vol_rhos.append(pearson_rho)


plt.semilogx(1/betas, los_rhos, label='LOS')
plt.semilogx(1/betas, vol_rhos, label='Volume')
plt.legend()
plt.show()
