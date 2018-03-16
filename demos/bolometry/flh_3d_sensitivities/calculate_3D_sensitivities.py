
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from raysect.core import Node, Point2D, Point3D, rotate_z, translate
from raysect.primitive import Mesh, import_stl, Cylinder, Subtract
from raysect.optical import World, UnityVolumeEmitter, UnitySurfaceEmitter
from raysect.optical.material import AbsorbingSurface

from cherab.aug.machine import plot_aug_wall_outline, import_mesh_segment, VESSEL, PSL, ICRH, DIVERTOR, A_B_COILS
from cherab.aug.bolometry import FDC_TUBE, FLX_TUBE, FVC_TUBE, FHS_TUBE, load_default_bolometer_config
from cherab.aug.bolometry import load_standard_inversion_grid
from cherab.tools.observers.inversion_grid import SensitivityMatrix


THETA_WIDTH = 0.5


def generate_annulus_mesh_segments(lower_corner, upper_corner, theta_width, theta_offset, world):

    material = UnityVolumeEmitter()

    # Set of points in x-z plane
    p1a = Point3D(lower_corner.x, 0, lower_corner.y)  # corresponds to lower corner is x-z plane
    p2a = Point3D(lower_corner.x, 0, upper_corner.y)
    p3a = Point3D(upper_corner.x, 0, upper_corner.y)  # corresponds to upper corner in x-z plane
    p4a = Point3D(upper_corner.x, 0, lower_corner.y)

    # Set of points rotated away from x-z plane
    p1b = p1a.transform(rotate_z(theta_width))
    p2b = p2a.transform(rotate_z(theta_width))
    p3b = p3a.transform(rotate_z(theta_width))
    p4b = p4a.transform(rotate_z(theta_width))

    vertices = [[p1a.x, p1a.y, p1a.z], [p2a.x, p2a.y, p2a.z],
                [p3a.x, p3a.y, p3a.z], [p4a.x, p4a.y, p4a.z],
                [p1b.x, p1b.y, p1b.z], [p2b.x, p2b.y, p2b.z],
                [p3b.x, p3b.y, p3b.z], [p4b.x, p4b.y, p4b.z]]

    triangles = [[1, 0, 3], [1, 3, 2],  # front face (x-z)
                 [7, 4, 5], [7, 5, 6],  # rear face (rotated out of x-z plane)
                 [5, 1, 2], [5, 2, 6],  # top face (x-y plane)
                 [3, 0, 4], [3, 4, 7],  # bottom face (x-y plane)
                 [4, 0, 5], [1, 5, 0],  # inner face (y-z plane)
                 [2, 3, 7], [2, 7, 6]]  # outer face (y-z plane)

    return Mesh(vertices=vertices, triangles=triangles, smoothing=False,
                transform=rotate_z(theta_offset), material=material, parent=world)


def calculate_sensitivity(grid, detector, theta_offset, world):

    volume_radiance_sensitivity = SensitivityMatrix(grid, detector.detector_id, 'Volume mean radiance sensitivity')
    volume_power_sensitivity = SensitivityMatrix(grid, detector.detector_id, 'Volume power sensitivity')

    for i in range(grid.count):

        p1, p2, p3, p4 = grid[i]

        segment = generate_annulus_mesh_segments(p1, p3, THETA_WIDTH, theta_offset, world)

        detector._volume_observer.observe()
        volume_radiance_sensitivity.sensitivity[i] = detector._volume_radiance_pipeline.value.mean
        volume_power_sensitivity.sensitivity[i] = detector._volume_power_pipeline.value.mean

        segment.parent = None

    return volume_radiance_sensitivity, volume_power_sensitivity


# Load default inversion grid
grid = load_standard_inversion_grid()


# FLH uses the full AUG mesh structure + box CAD file
full_world = World()
FULL_MESH_SEGMENTS = VESSEL + PSL + ICRH + DIVERTOR + A_B_COILS
import_mesh_segment(full_world, FULL_MESH_SEGMENTS)

# load detector box
flh_box = import_stl("/home/matt/CCFE/cadmesh/aug/diagnostics/bolometry/FLH_box_no_rfgrid.stl", parent=full_world,
                     material=AbsorbingSurface(), name="FLH_box", scaling=0.001)

# Load FLH detector description
flh = load_default_bolometer_config('FLH', parent=full_world)


# FLH slit centre coordinates
x = -0.209433
y = 1.0333
z = -0.465711
# Calculate angle range +/- 2.5 degrees around FLH slit centre angle
r = np.sqrt(x**2 + y**2)
slit_centre_angle = np.arctan2(y, x) / (2*np.pi) * 360
angle_range = np.arange(0, 5, THETA_WIDTH) - 2.5 + slit_centre_angle


flh2 = flh['FLH_A_CH2']
# remove the CSG slit from the scene since we are using CAD instead
flh2.slit.csg_aperture.parent = None


i = sys.argv[1]
angle = angle_range[i]
print("Starting angle {} - {:.4G} degrees".format(i, angle))
radiance_sensitivity, power_sensitivity = calculate_sensitivity(grid, flh2, angle, full_world)


data = {
    'angle': angle,
    'radiance_sensitivity': radiance_sensitivity.sensitivity,
    'power_sensitivity': power_sensitivity.sensitivity
}
pickle.dump(data, open('FLH2_3D_sensitivities_{}.pickle'.format(i), 'wb'))
