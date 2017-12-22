
import pickle
import numpy as np

from cherab.aug.bolometry import load_standard_inversion_grid, AUG_2D_TO_CHERAB_1D_GRID_MASK
from cherab.aug.bolometry.extras.standard_aug_grid import RECTILINEAR_GRID


# Config file describing which bolometer cameras are active
grid = load_standard_inversion_grid()


ny = 83
nx = 45
grid_index_2D_to_1D_map = {}
grid_index_1D_to_2D_map = {}
grid_laplacian = np.zeros((grid.count, grid.count))

unwrapped_cell_index = 0
for iy in range(ny):
    for ix in range(nx):

        _, p1, p2, p3, p4 = RECTILINEAR_GRID[iy][ix]

        for cell in grid:
            pc1, pc2, pc3, pc4 = cell

            if (pc1.x == p1.x and pc1.y == p1.y and
                pc2.x == p2.x and pc2.y == p2.y and
                pc3.x == p3.x and pc3.y == p3.y and
                pc4.x == p4.x and pc4.y == p4.y):

                grid_index_2D_to_1D_map[(ix, iy)] = unwrapped_cell_index
                grid_index_1D_to_2D_map[unwrapped_cell_index] = (ix, iy)
                unwrapped_cell_index += 1
                break

pickle.dump((grid_index_2D_to_1D_map, grid_index_1D_to_2D_map), open("grid_index_map.pickle", "wb"))

for ith_cell in range(grid.count):

    # get the 2D mesh coordinates of this cell
    ix, iy = grid_index_1D_to_2D_map[ith_cell]

    neighbours = 0

    try:
        n1 = grid_index_2D_to_1D_map[ix-1, iy]  # neighbour 1
        grid_laplacian[ith_cell, n1] = -1
        neighbours += 1
    except KeyError:
        pass

    try:
        n2 = grid_index_2D_to_1D_map[ix-1, iy+1]  # neighbour 2
        grid_laplacian[ith_cell, n2] = -1
        neighbours += 1
    except KeyError:
        pass

    try:
        n3 = grid_index_2D_to_1D_map[ix, iy+1]  # neighbour 3
        grid_laplacian[ith_cell, n3] = -1
        neighbours += 1
    except KeyError:
        pass

    try:
        n4 = grid_index_2D_to_1D_map[ix+1, iy+1]  # neighbour 4
        grid_laplacian[ith_cell, n4] = -1
        neighbours += 1
    except KeyError:
        pass

    try:
        n5 = grid_index_2D_to_1D_map[ix+1, iy]  # neighbour 5
        grid_laplacian[ith_cell, n5] = -1
        neighbours += 1
    except KeyError:
        pass

    try:
        n6 = grid_index_2D_to_1D_map[ix+1, iy-1]  # neighbour 6
        grid_laplacian[ith_cell, n6] = -1
        neighbours += 1
    except KeyError:
        pass

    try:
        n7 = grid_index_2D_to_1D_map[ix, iy-1]  # neighbour 7
        grid_laplacian[ith_cell, n7] = -1
        neighbours += 1
    except KeyError:
        pass

    try:
        n8 = grid_index_2D_to_1D_map[ix-1, iy-1]  # neighbour 8
        grid_laplacian[ith_cell, n8] = -1
        neighbours += 1
    except KeyError:
        pass

    grid_laplacian[ith_cell, ith_cell] = neighbours


np.save(open('grid_laplacian.ndarray', 'wb'), grid_laplacian)
