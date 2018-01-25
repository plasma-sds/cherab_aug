
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


# Make Latex available in matplotlib figures
rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})


phantom_results_fh = open(os.path.join(os.path.split(__file__)[0], 'inverted_phantoms.csv'), 'r')

lines = phantom_results_fh.readlines()

los_rhos = []
los_rhos_1000 = []
los_rhos_80 = []
vol_rhos = []
vol_rhos_1000 = []
vol_rhos_80 = []

for line in lines[1:]:
    row = [float(l) for l in line.split(',')]
    los_rhos.append(row[3])
    vol_rhos.append(row[9])
    los_rhos_1000.append(row[15])
    vol_rhos_1000.append(row[21])
    los_rhos_80.append(row[27])
    vol_rhos_80.append(row[33])

plt.ion()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(los_rhos_80, vol_rhos_80, 'C2.', label=r'$\beta_L = 0.0125$')
plt.plot(los_rhos_1000, vol_rhos_1000, 'C1.', label=r'$\beta_L = 0.001$')
plt.plot(los_rhos, vol_rhos, 'C0.', label=r'$\beta_L = 0$')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel(r'$\rho_c$ Sight-line method')
plt.ylabel(r'$\rho_c$ Volume ray-tracing')
plt.title('Performance of LOS VS ray-tracing volumes on phantoms')
plt.legend()


los_rhos = []
los_rhos_1000 = []
los_rhos_80 = []
vol_rhos = []
vol_rhos_1000 = []
vol_rhos_80 = []

for line in lines[1:]:
    row = [float(l) for l in line.split(',')]
    los_rhos.append(row[6])
    vol_rhos.append(row[12])
    los_rhos_1000.append(row[18])
    vol_rhos_1000.append(row[24])
    los_rhos_80.append(row[30])
    vol_rhos_80.append(row[36])

plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(los_rhos, vol_rhos, 'C0.', label=r'$\beta_L = 0$')
plt.plot(los_rhos_1000, vol_rhos_1000, 'C1.', label=r'$\beta_L = 0.001$')
plt.plot(los_rhos_80, vol_rhos_80, 'C2.', label=r'$\beta_L = 0.0125$')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel(r'$\rho_c$ Sight-line method')
plt.ylabel(r'$\rho_c$ Volume ray-tracing')
plt.title('Performance of LOS VS ray-tracing volumes on phantoms + Noise')
plt.legend()


#################################
# Start of radiated power plots #


def decide_power_range(phi_phantom):
    operation = 0
    if phi_phantom > 10:
        phi_phantom /= 2
        if phi_phantom > 5:
            operation = 1
        else:
            operation = 2
    elif phi_phantom < 5:
        operation = 3
    return operation


def perform_power_clamp(operation, phi):
    if operation == 1:
        phi /= 2
    elif operation == 2:
        phi = phi / 2 + 5
    elif operation == 3:
        phi += 5

    return phi


phantom_phi = []
los_phi = []
los_phi_1000 = []
los_phi_80 = []
vol_phi = []
vol_phi_1000 = []
vol_phi_80 = []

for line in lines[1:]:
    row = [float(l) for l in line.split(',')]

    operation = decide_power_range(row[1])
    phantom_phi.append(perform_power_clamp(operation, row[1]))
    los_phi.append(perform_power_clamp(operation, row[2]))
    vol_phi.append(perform_power_clamp(operation, row[8]))
    los_phi_1000.append(perform_power_clamp(operation, row[14]))
    vol_phi_1000.append(perform_power_clamp(operation, row[20]))
    los_phi_80.append(perform_power_clamp(operation, row[26]))
    vol_phi_80.append(perform_power_clamp(operation, row[32]))

plt.figure()
plt.plot([0, 15], [0, 15], 'k--')
plt.plot(phantom_phi, los_phi, 'C0.', label='LOS technique')
plt.plot(phantom_phi, vol_phi, 'C1.', label='Volume technique')
plt.xlim(5, 10)
plt.ylim(5, 10)
plt.xlabel(r'$\Phi_{rad}$ Phantom Power')
plt.ylabel(r'$\Phi_{rad}$ Volume ray-tracing')
plt.title(r'Inversions with \beta_L = 0')
plt.legend()

plt.figure()
plt.plot([0, 15], [0, 15], 'k--')
plt.plot(phantom_phi, los_phi_1000, 'C0.', label='LOS technique')
plt.plot(phantom_phi, vol_phi_1000, 'C1.', label='Volume technique')
plt.xlim(5, 10)
plt.ylim(5, 10)
plt.xlabel(r'$\Phi_{rad}$ Phantom Power')
plt.ylabel(r'$\Phi_{rad}$ Volume ray-tracing')
plt.title(r'Inversions with \beta_L = 0.001')
plt.legend()

plt.figure()
plt.plot([0, 15], [0, 15], 'k--')
plt.plot(phantom_phi, los_phi_80, 'C0.', label='LOS technique')
plt.plot(phantom_phi, vol_phi_80, 'C1.', label='Volume technique')
plt.xlim(5, 10)
plt.ylim(5, 10)
plt.xlabel(r'$\Phi_{rad}$ Phantom Power')
plt.ylabel(r'$\Phi_{rad}$ Volume ray-tracing')
plt.title(r'Inversions with \beta_L = 0.0125')
plt.legend()


phantom_phi = []
los_phi = []
los_phi_1000 = []
los_phi_80 = []
vol_phi = []
vol_phi_1000 = []
vol_phi_80 = []
los_phi_1000_error = []
vol_phi_1000_error = []

for line in lines[1:]:
    row = [float(l) for l in line.split(',')]

    operation = decide_power_range(row[1])
    phantom_phi.append(perform_power_clamp(operation, row[1]))
    los_phi.append(perform_power_clamp(operation, row[5]))
    vol_phi.append(perform_power_clamp(operation, row[11]))
    los_phi_1000.append(perform_power_clamp(operation, row[17]))
    vol_phi_1000.append(perform_power_clamp(operation, row[23]))
    los_phi_80.append(perform_power_clamp(operation, row[29]))
    vol_phi_80.append(perform_power_clamp(operation, row[35]))

    phip = perform_power_clamp(operation, row[1])
    phil = perform_power_clamp(operation, row[17])
    phiv = perform_power_clamp(operation, row[23])
    los_phi_1000_error.append(np.abs(phip - phil)/phip)
    vol_phi_1000_error.append(np.abs(phip - phiv)/phip)



plt.figure()
plt.plot([0, 15], [0, 15], 'k--')
plt.plot(phantom_phi, los_phi, 'C0.', label='LOS technique')
plt.plot(phantom_phi, vol_phi, 'C1.', label='Volume technique')
plt.xlim(5, 10)
plt.ylim(5, 10)
plt.xlabel(r'$\Phi_{rad}$ - Phantom Total Radiated Power (MW)')
plt.ylabel(r'$\Phi_{rad}$ - Inversion Total Radiated Power (MW)')
plt.title(r'Error in inverted total radiated power $\Phi_{rad}$ with $\beta_L = 0$')
plt.legend()

plt.figure()
plt.plot([0, 15], [0, 15], 'k--')
plt.plot(phantom_phi, los_phi_1000, 'C0.', label='LOS technique')
plt.plot(phantom_phi, vol_phi_1000, 'C1.', label='Volume technique')
plt.xlim(5, 10)
plt.ylim(5, 10)
plt.xlabel(r'$\Phi_{rad}$ - Phantom Total Radiated Power (MW)')
plt.ylabel(r'$\Phi_{rad}$ - Inversion Total Radiated Power (MW)')
plt.title(r'Error in inverted total radiated power $\Phi_{rad}$ with $\beta_L = 0.001$')
plt.legend()

plt.figure()
plt.plot([0, 15], [0, 15], 'k--')
plt.plot(phantom_phi, los_phi_80, 'C0.', label='LOS technique')
plt.plot(phantom_phi, vol_phi_80, 'C1.', label='Volume technique')
plt.xlim(5, 10)
plt.ylim(5, 10)
plt.xlabel(r'$\Phi_{rad}$ - Phantom Total Radiated Power (MW)')
plt.ylabel(r'$\Phi_{rad}$ - Inversion Total Radiated Power (MW)')
plt.title(r'Error in inverted total radiated power $\Phi_{rad}$ with $\beta_L = 0.0125$')
plt.legend()


print()
print("Los phi distribution", np.mean(los_phi_1000_error), np.std(los_phi_1000_error))
print("Vol phi distribution", np.mean(vol_phi_1000_error), np.std(vol_phi_1000_error))
