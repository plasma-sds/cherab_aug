
import os
import re

from cherab.tools.observers.inversion_grid import EmissivityGrid
from cherab.aug.bolometry import AUG_2D_TO_CHERAB_1D_GRID_MASK


# replace with path to your BLB geometry directory
AUG_BLB_CASE_DIRECTORY = os.path.expanduser("~/CCFE/mst1/aug_bolometry/BLB.33280_2")

GRID_DIMENSIONS = "^Grid points:\s*([0-9]+)x([0-9]+)$"
USED_CAMERAS = "^Used cameras:\s*((?:[A-Z]{3}\s?)*)"


def load_blb_config_file(file_path):

    directory, name = os.path.split(file_path)

    # load config file
    fh = open(os.path.join(directory, "config"), "r")
    config_file = fh.readlines()
    fh.close()

    for line in config_file:
        match = re.match(GRID_DIMENSIONS, line)
        if match:
            nx = int(match.group(1))
            ny = int(match.group(2))
            if not (nx == 45 and ny == 83):
                raise ValueError("Expected AUG standard grid size is nx=45 and ny=83.")
            break
    else:
        raise ValueError("Bolometry config file did not contain a 'Grid points: NxM' specification.")

    for line in config_file:
        match = re.match(USED_CAMERAS, line)
        if match:
            active_cameras = match.group(1).split()
            for camera in active_cameras:
                if camera not in ['FDC', 'FHC', 'FHS', 'FLH', 'FLX', 'FVC']:
                    raise ValueError("Sensitivity matrix for camera '{}' is not available.".format(camera))
            break
    else:
        raise ValueError("Bolometry config file did not contain a 'Used cameras: FHC FVC...' specification.")

    return active_cameras


def load_blb_result_file(file_path, grid):

    # case ID will be the file name without the file extension code
    case_id = os.path.splitext(os.path.split(file_path)[1])[0]

    nx = 45
    ny = 83

    # load result file
    fh = open(file_path, "r")
    results_file = fh.readlines()
    fh.close()

    start_of_grid_values = 3+nx+1+ny+1
    start_of_detector_results = start_of_grid_values + (nx + 1)*(ny + 1)
    header = results_file[0:3]
    grid_definition = results_file[3:start_of_grid_values]
    raw_grid_values = results_file[start_of_grid_values:start_of_detector_results]
    raw_detector_values = results_file[start_of_detector_results:]

    emissivity_values = []
    for iy in range(ny):
        for ix in range(nx):
            if AUG_2D_TO_CHERAB_1D_GRID_MASK[iy, ix]:

                # calculate the AUG result data indices for the corners of this cell
                i1 = iy + ix * (ny+1)
                i2 = (iy+1) + ix * (ny+1)
                i3 = (iy+1) + (ix+1) * (ny+1)
                i4 = iy + (ix+1) * (ny+1)

                # calculate the average emissivity over the cell and save it
                avg_emissivity = (float(raw_grid_values[i1].strip()) + float(raw_grid_values[i2].strip()) +
                                  float(raw_grid_values[i3].strip()) + float(raw_grid_values[i4].strip())) / 4
                emissivity_values.append(avg_emissivity)

    return EmissivityGrid(grid, case_id=case_id, emissivities=emissivity_values)

