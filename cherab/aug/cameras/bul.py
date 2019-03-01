
import os
import csv
import numpy as np
import xml.etree.ElementTree as etree
from raysect.core import Point2D, Point3D, Vector3D
from raysect.optical.observer import PowerPipeline2D, VectorCamera

from cherab.tools.inversions import ToroidalVoxelGrid


def load_bul_voxel_grid(parent=None, name=None, active=None):

    directory = os.path.split(__file__)[0]
    voxel_grid_file = os.path.join(directory, "bul_voxel_grid.csv")

    voxel_coordinates = []
    with open(voxel_grid_file, 'r') as fh:
        reader = csv.reader(fh, )

        for row in reader:
            if row[0][0] == '#':
                continue
            v1 = Point2D(float(row[1]), float(row[2]))
            v2 = Point2D(float(row[3]), float(row[4]))
            v3 = Point2D(float(row[5]), float(row[6]))
            v4 = Point2D(float(row[7]), float(row[8]))
            voxel_coordinates.append((v1, v2, v3, v4))

    voxel_grid = ToroidalVoxelGrid(voxel_coordinates, parent=parent, name=name, active=active)

    return voxel_grid


def load_bul_camera(camera='06Bul3', shot=35050, parent=None, pipelines=None, stride=1):

    directory = os.path.split(__file__)[0]

    if camera == '06Bul3':
        if shot >= 35050:
            camera_config = _load_camera_calibration(os.path.join(directory, './data/06Bul3_los.xml'), interp_factor=30)
        else:
            camera_config = _load_camera_calibration(os.path.join(directory, './data/06Bul3_los_old.xml'), interp_factor=30)

    elif camera == '06Bul':
        camera_config = _load_camera_calibration(os.path.join(directory, './data/06Bul_los.xml'), interp_factor=30)

    else:
        raise ValueError("Unidentified camera - '{}'".format(camera))

    if not pipelines:
        power_unfiltered = PowerPipeline2D(display_unsaturated_fraction=0.96, name="Unfiltered Power (W)")
        power_unfiltered.display_update_time = 15
        pipelines = [power_unfiltered]

    pixels_shape, pixel_origins, pixel_directions = camera_config
    camera = VectorCamera(pixel_origins[::stride, ::stride], pixel_directions[::stride, ::stride],
                          pipelines=pipelines, parent=parent)
    camera.spectral_bins = 15
    camera.pixel_samples = 1

    return camera


def _load_camera_calibration(filepath, interp_factor=15):

    tree = etree.parse(filepath)
    root = tree.getroot()

    nx = int(root.attrib['Nx'])
    ny = int(root.attrib['Ny'])
    raw_shape = (nx, ny)
    print("raw_shape", raw_shape)
    camera_shape = (nx + (nx-1)*(interp_factor-1), ny + (ny-1)*(interp_factor-1))
    print("camera_shape", camera_shape)

    origin = Point3D(float(root.attrib['eyeX']), float(root.attrib['eyeY']), float(root.attrib['eyeZ']))
    # centre_dir = Vector3D(1.76114249229-origin.x, 0.975705087185-origin.y, -0.485737383366-origin.z)
    # centre_dir.normalise()

    pixel_origins = np.empty(shape=camera_shape, dtype=np.dtype(object))
    raw_directions = np.empty(shape=raw_shape, dtype=np.dtype(object))
    raw_lengths = np.empty(shape=(ny, nx))
    pixel_directions = np.empty(shape=camera_shape, dtype=np.dtype(object))

    for child in root:

        losdict = child.attrib

        ix = int(losdict['ix'])
        iy = ny - 1 - int(losdict['iy'])

        x = float(losdict['x'])
        y = float(losdict['y'])
        z = float(losdict['z'])

        direction = Vector3D(x - origin.x, y - origin.y, z - origin.z)
        raw_lengths[iy, ix] = direction.length
        raw_directions[ix, iy] = direction.normalise()

    for ni in range(camera_shape[0]):
        for nj in range(camera_shape[1]):

            x_remainder = ni % interp_factor
            y_remainder = nj % interp_factor

            # percentage distance between raw vectors
            x_percent = x_remainder / interp_factor
            y_percent = y_remainder / interp_factor

            # effective indicies of nearest corner
            effi = int((ni-x_remainder)/interp_factor)
            effj = int((nj-y_remainder)/interp_factor)
            corner_vec = raw_directions[effi, effj]
            try:
                next_xvec = raw_directions[effi + 1, effj]
            except IndexError:
                next_xvec = raw_directions[effi, effj]
            try:
                next_yvec = raw_directions[effi, effj + 1]
            except IndexError:
                next_yvec = raw_directions[effi, effj]

            # direction is corner A + relative percentages of (B-A) and (C-A)
            direction = corner_vec + (next_xvec - corner_vec)*x_percent + (next_yvec - corner_vec)*y_percent
            pixel_directions[ni, nj] = direction

    # moving camera forward because of mesh collision
    # print("Origin {}".format(origin))
    # print("Centre dir {}".format(centre_dir))
    # print("Direction length {}".format(centre_dir.length))
    # origin = origin + 0.35 * centre_dir
    # print("new origin {}".format(origin))

    for ix in range(camera_shape[0]):
        for iy in range(camera_shape[1]):
            pixel_origins[ix, iy] = origin.copy()

    return camera_shape, pixel_origins, pixel_directions
