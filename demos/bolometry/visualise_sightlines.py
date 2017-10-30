
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from raysect.core import World
from raysect.core.ray import Ray as CoreRay

from cherab.aug.machine import plot_aug_wall_outline, import_mesh_segment, VESSEL, PSL, ICRH, DIVERTOR, A_B_COILS
from cherab.aug.bolometry import FDC_TUBE, FLX_TUBE, FVC_TUBE, FHS_TUBE, AUG_FOIL_BOLOMETER_DETECTORS,\
    FDC_FOILS, FHC1_FOILS, FHC2_FOILS, FHC3_FOILS,\
    FHS_FOILS, FLH_FOILS, FLX_FOILS, FVC1_FOILS, FVC2_FOILS


def plot_detectors(detector_dictionary, title, world):

    plt.figure()
    plot_aug_wall_outline()

    for detector_name, detector in sorted(detector_dictionary.items()):
        # unpack detector configuration
        pinhole_name, centre_point, normal_vec, basis_x, dx, basis_y, dy = detector

        # Find the next intersection point of the ray with the world
        intersection = world.hit(CoreRay(centre_point, normal_vec))
        if intersection is not None:
            hit_point = intersection.hit_point.transform(intersection.primitive_to_world)
        else:
            hit_point = centre_point + normal_vec * 2.0

        # Traverse the ray with equation for a parametric line,
        # i.e. t=0->1 traverses the ray path.
        parametric_vector = centre_point.vector_to(hit_point)
        t_samples = np.arange(0, 1, 0.01)

        # Setup some containers for useful parameters along the ray trajectory
        ray_r_points = []
        ray_z_points = []

        # At each ray position sample the parameters of interest.
        for i, t in enumerate(t_samples):
            # Get new sample point location and log distance
            x = centre_point.x + parametric_vector.x * t
            y = centre_point.y + parametric_vector.y * t
            z = centre_point.z + parametric_vector.z * t
            ray_r_points.append(np.sqrt(x**2 + y**2))
            ray_z_points.append(z)

        plt.plot(ray_r_points, ray_z_points, 'k')
        plt.plot(ray_r_points[0], ray_z_points[0], 'b.')
        plt.plot(ray_r_points[-1], ray_z_points[-1], 'r.')

    plt.title(title)


full_world = World()
FULL_MESH_SEGMENTS = VESSEL + PSL + ICRH + DIVERTOR + A_B_COILS
import_mesh_segment(full_world, FULL_MESH_SEGMENTS)

fhc = FHC1_FOILS.copy()
fhc.update(FHC2_FOILS)
fhc.update(FHC3_FOILS)

fvc = FVC1_FOILS.copy()
fvc.update(FVC2_FOILS)

plot_detectors(fhc, 'FHC', full_world)
plot_detectors(FLH_FOILS, 'FLH', full_world)

fhs_world = World()
import_mesh_segment(fhs_world, FHS_TUBE)
plot_detectors(FHS_FOILS, 'FHS', fhs_world)

fvc_world = World()
import_mesh_segment(fvc_world, FVC_TUBE)
plot_detectors(fvc, 'FVC', fvc_world)

fdc_world = World()
import_mesh_segment(fdc_world, FDC_TUBE)
plot_detectors(FDC_FOILS, 'FDC', fdc_world)

flx_world = World()
import_mesh_segment(flx_world, FLX_TUBE)
plot_detectors(FLX_FOILS, 'FLX', flx_world)
