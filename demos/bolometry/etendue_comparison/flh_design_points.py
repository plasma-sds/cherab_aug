
import re
import numpy as np
import pickle
import matplotlib.pyplot as plt

from raysect.core import translate, rotate_basis, Point3D, Vector3D, Ray as CoreRay
from raysect.primitive import Box, Subtract
from raysect.optical import World
from raysect.optical.material import AbsorbingSurface, NullMaterial
from raysect.core.math.sampler import DiskSampler3D, RectangleSampler3D, TargettedHemisphereSampler

from cherab.aug.bolometry import FDC_TUBE, FLX_TUBE, FVC_TUBE, FHS_TUBE, load_default_bolometer_config
from cherab.aug.bolometry import load_standard_inversion_grid


R_2_PI = 1 / (2 * np.pi)


def analytic_etendue(area_det, area_slit, distance, alpha, gamma):

    return area_det * area_slit * np.cos(alpha/360 * (2*np.pi)) * np.cos(gamma/360 * (2*np.pi)) / distance**2


def raytraced_etendue(distance, point_sampler, detector_area, ray_count=100000, batches=10):

    # generate the transform to the detector position and orientation
    detector_transform = translate(0, 0, distance) * rotate_basis(Vector3D(0, 0, -1), Vector3D(0, -1, 0))

    # generate bounding sphere and convert to local coordinate system
    sphere = hole.bounding_sphere()
    spheres = [(sphere.centre.transform(detector_transform), sphere.radius, 1.0)]

    # instance targetted pixel sampler
    targetted_sampler = TargettedHemisphereSampler(spheres)

    solid_angle = 2 * np.pi
    etendue_sampled = solid_angle * detector_area

    etendues = []
    for i in range(batches):

        # sample pixel origins
        origins = point_sampler(samples=ray_count)

        passed = 0.0
        for origin in origins:

            # obtain targetted vector sample
            direction, pdf = targetted_sampler(origin, pdf=True)
            while pdf == 0.0:
                direction, pdf = targetted_sampler(origin, pdf=True)
            path_weight = R_2_PI * direction.z/pdf

            origin = origin.transform(detector_transform)
            direction = direction.transform(detector_transform)

            while True:

                # Find the next intersection point of the ray with the world
                intersection = world.hit(CoreRay(origin, direction))

                if intersection is None:
                    passed += 1 * path_weight
                    break

                elif isinstance(intersection.primitive.material, NullMaterial):
                    hit_point = intersection.hit_point.transform(intersection.primitive_to_world)
                    origin = hit_point + direction * 1E-9
                    continue

                else:
                    break

        if passed == 0:
            raise ValueError("Something is wrong with the scene-graph, calculated etendue should not zero.")

        etendue_fraction = passed / ray_count

        etendues.append(etendue_sampled * etendue_fraction)

    etendue = np.mean(etendues)
    etendue_error = np.std(etendues)

    return etendue, etendue_error


###########################

world = World()
flh = load_default_bolometer_config('FLH', parent=world)

for detector in flh:

    print()
    print(detector.detector_id)

    pinhole = detector.slit
    detector_to_pinhole = detector.centre_point.vector_to(pinhole.centre_point).normalise()

    # Setup detector parameters
    print("Detector")
    det_xwidth = detector.dx
    det_ywidth = detector.dy
    area_detector = det_xwidth * det_ywidth
    d = min(det_xwidth, det_ywidth)
    dxd_factor = max(detector.dx / detector.dy, detector.dy / detector.dx)
    cos_theta = detector_to_pinhole.dot(detector.normal_vec)
    theta = np.rad2deg(np.arccos(np.abs(cos_theta)))
    print("dx: {:.4G}".format(detector.dx), "dy: {:.4G}".format(detector.dy), "dxd-factor: {:.4G}".format(dxd_factor))
    print("theta: {:.3G}".format(theta), "cos_theta: {:.4G}".format(cos_theta))

    # Setup pinhole parameters
    print("Pinhole")
    ph_xwidth = pinhole.dx
    ph_ywidth = pinhole.dy
    area_pinhole = ph_xwidth * ph_ywidth
    ph_dxd_factor = max(ph_xwidth / ph_ywidth, ph_ywidth / ph_xwidth)
    ph_normal = pinhole.basis_x.cross(pinhole.basis_y).normalise()
    cos_phi = detector_to_pinhole.dot(ph_normal)
    phi = np.rad2deg(np.arccos(np.abs(cos_phi)))
    print("dx: {:.4G}".format(ph_xwidth), "dy: {:.4G}".format(ph_ywidth), "dxd-factor: {:.4G}".format(ph_dxd_factor))
    print("phi: {:.3G}".format(phi), "cos_phi: {:.4G}".format(cos_phi))

    separation = detector.centre_point.distance_to(detector.slit.centre_point)
    normalised_separation = separation/d
    print("Distances")
    print("separation: {:.4G}".format(separation), "normalised separation: {:.4G}".format(normalised_separation))


    point_sampler = RectangleSampler3D(width=det_xwidth, height=det_ywidth)
    target_plane = Box(Point3D(-100000, -100000, -0.000001), Point3D(100000, 100000, 0.000001))
    hole = Box(Point3D(-det_xwidth/2, -det_ywidth/2, -0.00001), Point3D(det_xwidth/2, det_ywidth/2, 0.00001), parent=world, material=NullMaterial())
    pinhole = Subtract(target_plane, hole, parent=world, material=AbsorbingSurface())

    analytic_value = analytic_etendue(area_detector, area_pinhole, separation, theta, phi)

    raytraced_value, raytraced_error = raytraced_etendue(separation, point_sampler, area_detector)

    detector.calculate_etendue(ray_count=100000)
    cherab_auto_etendue = detector.etendue

    print("Results")
    print("Analytic etendue: {:.4G}".format(analytic_value))
    print("Manual Ray-traced etendue: {:.4G} +- {:.4G}".format(raytraced_value, raytraced_error))
    print("Auto Ray-traced etendue: {:.4G}".format(cherab_auto_etendue))
    print("Relative error: {:.4G}".format(np.abs(cherab_auto_etendue-analytic_value)/cherab_auto_etendue))

    hole.parent = None
    pinhole.parent = None

