
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
# ax.legend(handles[::-1], labels[::-1])


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
plt.plot(los_rhos_80, vol_rhos_80, 'C2.', label=r'$\beta_L = 0.0125$')
plt.plot(los_rhos_1000, vol_rhos_1000, 'C1.', label=r'$\beta_L = 0.001$')
plt.plot(los_rhos, vol_rhos, 'C0.', label=r'$\beta_L = 0$')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel(r'$\rho_c$ Sight-line method')
plt.ylabel(r'$\rho_c$ Volume ray-tracing')
plt.title('Performance of LOS VS ray-tracing volumes on phantoms + Noise')
plt.legend()




# Looking at convergence rate. Turns out volumes take longer so ignore this one.
# print()
# print(np.mean(los_iter), np.std(los_iter))
# print(np.mean(vol_iter), np.std(vol_iter))
# print(np.mean(los_iter_1000), np.std(los_iter_1000))
# print(np.mean(vol_iter_1000), np.std(vol_iter_1000))
