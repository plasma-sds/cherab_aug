
import csv
import os
import numpy as np
import scipy
import matplotlib.pyplot as plt

from cherab.tools.inversions import invert_sart, invert_constrained_sart
from cherab.tools.observers.inversion_grid import EmissivityGrid
from cherab.tools.observers.bolometry import assemble_weight_matrix
from cherab.aug.bolometry import load_emissivity_phantom, load_standard_inversion_grid, load_default_bolometer_config


NOISE_VARIANCE = 0.05
OUTPUT_DIR = os.path.join(os.path.split(__file__)[0], "output_data")

ofile = open("inverted_phantoms.csv", "w")
writer = csv.writer(ofile, delimiter=',', quotechar='"')
header = ["Phantom ID", "Phantom Power",
          "SART LOS Power (beta=0)", "SART LOS corr", "iter", "SART Volume Power (beta=0)", "SART Volume corr", "iter",
          "C-SART LOS Power  (beta=1000)", "C-SART LOS corr", "iter", "C-SART Volume Power  (beta=1000)", "C-SART Volume corr", "iter",
          "C-SART LOS Power  (beta=80)", "C-SART LOS corr", "iter", "C-SART Volume Power  (beta=80)", "C-SART Volume corr", "iter"]
writer.writerow(header)

# 'FVC1_A_CH1', 'FVC2_E_CH17', 'FLX_A_CH1', 'FHS_A_CH1'
EXCLUDED_CHANNELS = ['FDC_A_CH1', 'FDC_G_CH28', 'FHC1_A_CH1', 'FHC1_A_CH2', 'FHC1_A_CH3']

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


detector_keys, los_weight_matrix, vol_weight_matrix = assemble_weight_matrix([fhc, flh, fhs, fvc, fdc, flx],
                                                                             excluded_detectors=EXCLUDED_CHANNELS)

# invert all 94 phantoms
for i in range(94):

    row = []

    phantom_index = i + 1
    phantom_id = 'AUG_emission_phantom_' + str(phantom_index).zfill(3)
    emiss_phantom = load_emissivity_phantom(phantom_id)
    emissivities = emiss_phantom.emissivities
    row.append(phantom_index)
    row.append(float('{:.4G}'.format(emiss_phantom.total_radiated_power()/1E6)))

    emiss_phantom.plot()
    plt.axis('equal')
    plt.savefig(os.path.join(OUTPUT_DIR, "ph" + str(phantom_index).zfill(2) + "_phantom_emissivity.png"))

    # calculate observed power - uses volume method only since this is realistic
    observed_power = np.dot(vol_weight_matrix, emissivities)
    obs_with_noise = observed_power * (np.random.randn(len(observed_power)) * NOISE_VARIANCE + 1)

    inverted_emiss_vector, conv = invert_sart(los_weight_matrix, observed_power, max_iterations=150)
    inverted_emiss = EmissivityGrid(grid, case_id='Phantom {} - LOS SART method'.format(str(phantom_index).zfill(2)), emissivities=inverted_emiss_vector)
    inverted_emiss.plot()
    plt.axis('equal')
    plt.savefig(os.path.join(OUTPUT_DIR, "ph" + str(phantom_index).zfill(2) + "_sart_los.png"))
    row.append(float('{:.4G}'.format(inverted_emiss.total_radiated_power()/1E6)))
    row.append(float('{:.3G}'.format(np.corrcoef(emiss_phantom.emissivities, inverted_emiss.emissivities)[0][1])))
    row.append(len(conv))

    inverted_emiss_vector, conv = invert_sart(los_weight_matrix, obs_with_noise, max_iterations=150)
    inverted_emiss = EmissivityGrid(grid, case_id='Phantom {} - LOS SART + Noise method'.format(str(phantom_index).zfill(2)), emissivities=inverted_emiss_vector)
    row.append(float('{:.4G}'.format(inverted_emiss.total_radiated_power()/1E6)))
    row.append(float('{:.3G}'.format(np.corrcoef(emiss_phantom.emissivities, inverted_emiss.emissivities)[0][1])))
    row.append(len(conv))

    inverted_emiss_vector, conv = invert_sart(vol_weight_matrix, observed_power, max_iterations=150)
    inverted_emiss = EmissivityGrid(grid, case_id='Phantom {} - VOL SART method'.format(str(phantom_index).zfill(2)), emissivities=inverted_emiss_vector)
    inverted_emiss.plot()
    plt.axis('equal')
    plt.savefig(os.path.join(OUTPUT_DIR, "ph" + str(phantom_index).zfill(2) + "_sart_volume.png"))
    row.append(float('{:.4G}'.format(inverted_emiss.total_radiated_power()/1E6)))
    row.append(float('{:.3G}'.format(np.corrcoef(emiss_phantom.emissivities, inverted_emiss.emissivities)[0][1])))
    row.append(len(conv))

    inverted_emiss_vector, conv = invert_sart(vol_weight_matrix, obs_with_noise, max_iterations=150)
    inverted_emiss = EmissivityGrid(grid, case_id='Phantom {} - VOL SART + Noise method'.format(str(phantom_index).zfill(2)), emissivities=inverted_emiss_vector)
    row.append(float('{:.4G}'.format(inverted_emiss.total_radiated_power()/1E6)))
    row.append(float('{:.3G}'.format(np.corrcoef(emiss_phantom.emissivities, inverted_emiss.emissivities)[0][1])))
    row.append(len(conv))

    inverted_emiss_vector, conv = invert_constrained_sart(los_weight_matrix, GRID_LAPLACIAN, observed_power, max_iterations=150, beta_laplace=1000)
    inverted_emiss = EmissivityGrid(grid, case_id='Phantom {} - CSART LOS method'.format(str(phantom_index).zfill(2)), emissivities=np.squeeze(np.asarray(inverted_emiss_vector)))
    row.append(float('{:.4G}'.format(inverted_emiss.total_radiated_power()/1E6)))
    row.append(float('{:.3G}'.format(np.corrcoef(emiss_phantom.emissivities, inverted_emiss.emissivities)[0][1])))
    row.append(len(conv))

    inverted_emiss_vector, conv = invert_constrained_sart(los_weight_matrix, GRID_LAPLACIAN, obs_with_noise, max_iterations=150, beta_laplace=1000)
    inverted_emiss = EmissivityGrid(grid, case_id='Phantom {} - CSART LOS + Noise method'.format(str(phantom_index).zfill(2)), emissivities=np.squeeze(np.asarray(inverted_emiss_vector)))
    row.append(float('{:.4G}'.format(inverted_emiss.total_radiated_power()/1E6)))
    row.append(float('{:.3G}'.format(np.corrcoef(emiss_phantom.emissivities, inverted_emiss.emissivities)[0][1])))
    row.append(len(conv))

    inverted_emiss_vector, conv = invert_constrained_sart(vol_weight_matrix, GRID_LAPLACIAN, observed_power, max_iterations=150, beta_laplace=1000)
    inverted_emiss = EmissivityGrid(grid, case_id='Phantom {} - CSART Volume method'.format(str(phantom_index).zfill(2)), emissivities=np.squeeze(np.asarray(inverted_emiss_vector)))
    row.append(float('{:.4G}'.format(inverted_emiss.total_radiated_power()/1E6)))
    row.append(float('{:.3G}'.format(np.corrcoef(emiss_phantom.emissivities, inverted_emiss.emissivities)[0][1])))
    row.append(len(conv))

    inverted_emiss_vector, conv = invert_constrained_sart(vol_weight_matrix, GRID_LAPLACIAN, obs_with_noise, max_iterations=150, beta_laplace=1000)
    inverted_emiss = EmissivityGrid(grid, case_id='Phantom {} - CSART Volume + Noise method'.format(str(phantom_index).zfill(2)), emissivities=np.squeeze(np.asarray(inverted_emiss_vector)))
    row.append(float('{:.4G}'.format(inverted_emiss.total_radiated_power()/1E6)))
    row.append(float('{:.3G}'.format(np.corrcoef(emiss_phantom.emissivities, inverted_emiss.emissivities)[0][1])))
    row.append(len(conv))

    inverted_emiss_vector, conv = invert_constrained_sart(los_weight_matrix, GRID_LAPLACIAN, observed_power, max_iterations=150, beta_laplace=80)
    inverted_emiss = EmissivityGrid(grid, case_id='Phantom {} - CSART LOS method'.format(str(phantom_index).zfill(2)), emissivities=np.squeeze(np.asarray(inverted_emiss_vector)))
    inverted_emiss.plot()
    plt.axis('equal')
    plt.savefig(os.path.join(OUTPUT_DIR, "ph" + str(phantom_index).zfill(2) + "_csart_los.png"))
    row.append(float('{:.4G}'.format(inverted_emiss.total_radiated_power()/1E6)))
    row.append(float('{:.3G}'.format(np.corrcoef(emiss_phantom.emissivities, inverted_emiss.emissivities)[0][1])))
    row.append(len(conv))

    inverted_emiss_vector, conv = invert_constrained_sart(los_weight_matrix, GRID_LAPLACIAN, obs_with_noise, max_iterations=150, beta_laplace=80)
    inverted_emiss = EmissivityGrid(grid, case_id='Phantom {} - CSART LOS + Noise method'.format(str(phantom_index).zfill(2)), emissivities=np.squeeze(np.asarray(inverted_emiss_vector)))
    row.append(float('{:.4G}'.format(inverted_emiss.total_radiated_power()/1E6)))
    row.append(float('{:.3G}'.format(np.corrcoef(emiss_phantom.emissivities, inverted_emiss.emissivities)[0][1])))
    row.append(len(conv))

    inverted_emiss_vector, conv = invert_constrained_sart(vol_weight_matrix, GRID_LAPLACIAN, observed_power, max_iterations=150, beta_laplace=80)
    inverted_emiss = EmissivityGrid(grid, case_id='Phantom {} - CSART Volume method'.format(str(phantom_index).zfill(2)), emissivities=np.squeeze(np.asarray(inverted_emiss_vector)))
    inverted_emiss.plot()
    plt.axis('equal')
    plt.savefig(os.path.join(OUTPUT_DIR, "ph" + str(phantom_index).zfill(2) + "_csart_volume.png"))
    row.append(float('{:.4G}'.format(inverted_emiss.total_radiated_power()/1E6)))
    row.append(float('{:.3G}'.format(np.corrcoef(emiss_phantom.emissivities, inverted_emiss.emissivities)[0][1])))
    row.append(len(conv))

    inverted_emiss_vector, conv = invert_constrained_sart(vol_weight_matrix, GRID_LAPLACIAN, obs_with_noise, max_iterations=150, beta_laplace=80)
    inverted_emiss = EmissivityGrid(grid, case_id='Phantom {} - CSART Volume + Noise method'.format(str(phantom_index).zfill(2)), emissivities=np.squeeze(np.asarray(inverted_emiss_vector)))
    row.append(float('{:.4G}'.format(inverted_emiss.total_radiated_power()/1E6)))
    row.append(float('{:.3G}'.format(np.corrcoef(emiss_phantom.emissivities, inverted_emiss.emissivities)[0][1])))
    row.append(len(conv))

    plt.close("all")
    writer.writerow(row)
    print(row)

ofile.close()
