
import numpy as np

from raysect.optical import World

from cherab.aug.bolometry import FDC_TUBE, FLX_TUBE, FVC_TUBE, FHS_TUBE, load_default_bolometer_config
from cherab.aug.bolometry import load_standard_inversion_grid

# Load default inversion grid
grid = load_standard_inversion_grid()


fvc_world = World()
fvc = load_default_bolometer_config('FVC', parent=fvc_world)

fvc_ch10 = fvc['FVC1_C_CH10']
detector = fvc_ch10
detector.calculate_etendue()

mb_area_foil = 4.940000E-06  # f_foil
mb_area_pinhole = 1.811040E-05  # f_blende
mb_d = 0.07304400  # d
mb_delta = -8.400000  # delta
mb_gamma = 1.993000  # gamma
mb_alpha = -6.407000  # alpha
mb_etendue = 1.325230E-09

mb_mesh_area_reduction = mb_area_pinhole/(detector.slit.dx * detector.slit.dy)
mb_omega = mb_area_pinhole * np.cos(np.deg2rad(mb_gamma)) / mb_d**2
mb_calc_etendue = np.cos(np.deg2rad(mb_alpha)) * mb_area_foil * mb_omega

cherab_solid_angle = detector.etendue / detector._volume_observer.etendue * 2*np.pi


print()
print('MB foil area            {:.4G} m^2'.format(mb_area_foil))
print('CHERAB foil area        {:.4G} m^2'.format(detector.dx * detector.dy))

print()
print('MB pinhole area         {:.4G} m^2'.format(mb_area_pinhole))
print('CHERAB pinhole area     {:.4G} m^2'.format(detector.slit.dx * detector.slit.dy))
print('Area reduction A_frac   {:.4G}'.format(mb_mesh_area_reduction))

print()
print('MB p-a distance         {:.4G} m'.format(mb_d))
print('CHERAB p-a distance     {:.4G} m'.format(detector.centre_point.distance_to(detector.slit.centre_point)))

print()
print('MB solid angle          {:.4G} str'.format(mb_omega))
print('CHERAB solid angle      {:.4G} str'.format(cherab_solid_angle))
print('MB solid angle frac     {:.4G}'.format(mb_omega / (2 * np.pi)))
print('CHERAB solid angle frac {:.4G}'.format(detector.etendue / detector._volume_observer.etendue))


print()
print('MB Etendue              {:.4G} m^2 str'.format(mb_etendue * 4 * np.pi))
print('Formula with MB numbers {:.4G} m^2 str'.format(mb_calc_etendue))
print('CHERAB etendue          {:.4G} m^2 str'.format(detector.etendue))
print('CHERAB etendue * A_frac {:.4G} m^2 str'.format(detector.etendue * mb_mesh_area_reduction))
