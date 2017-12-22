

import numpy as np
from cherab.aug.bolometry import load_standard_inversion_grid
from cherab.aug.bolometry.standard_aug_grid import RECTILINEAR_GRID

# Config file describing which bolometer cameras are active
grid = load_standard_inversion_grid()


ny = 83
nx = 45
grid_mask = np.empty((ny, nx), dtype=bool)

for iy in range(ny):
    for ix in range(nx):

        _, p1, p2, p3, p4 = RECTILINEAR_GRID[iy][ix]

        for cell in grid:
            pc1, pc2, pc3, pc4 = cell

            if (pc1.x == p1.x and pc1.y == p1.y and
                pc2.x == p2.x and pc2.y == p2.y and
                pc3.x == p3.x and pc3.y == p3.y and
                pc4.x == p4.x and pc4.y == p4.y):
                grid_mask[iy, ix] = True
                break
        else:
            grid_mask[iy, ix] = False

np.save(open("grid_mask.ndarray", "wb"), grid_mask)

i = 0
for iy in range(ny):
    for ix in range(nx):
        if grid_mask[iy, ix] == True:
            i += 1

print("cells found = {}".format(i))
