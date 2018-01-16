
from raysect.optical import World

from cherab.core.atomic import carbon
from cherab.core.model import TotalRadiatedPower
from cherab.solps import load_solps_from_raw_output
from cherab.openadas import OpenADAS
from cherab.aug.bolometry import load_default_bolometer_config
from cherab.aug.bolometry import load_standard_inversion_grid


world = World()
adas = OpenADAS(permit_extrapolation=True)  # create atomic data source


# Change to your local solps output files directory
sim = load_solps_from_raw_output('/home/mcarr/mst1/aug_2016/solps_testcase', debug=True)

mesh = sim.mesh
plasma = sim.create_plasma()
plasma.parent = world
plasma.atomic_data = adas


grid = load_standard_inversion_grid()
flx = load_default_bolometer_config('FLX', parent=world, inversion_grid=grid)
fdc = load_default_bolometer_config('FDC', parent=world, inversion_grid=grid)
fvc = load_default_bolometer_config('FVC', parent=world, inversion_grid=grid)
fhs = load_default_bolometer_config('FHS', parent=world, inversion_grid=grid)
fhc = load_default_bolometer_config('FHC', parent=world, inversion_grid=grid)
flh = load_default_bolometer_config('FLH', parent=world, inversion_grid=grid)


for species in plasma.composition:
    print(species.element.name, species.ionisation)
    # Don't want to add fully ionised species, since they don't radiate
    if species.ionisation != species.element.atomic_number:
        plasma.models.add(TotalRadiatedPower(species.element, species.ionisation))


for foil in flx:
    power_observed = foil.observe_volume_power()
    print('{} - {}W'.format(foil.detector_id, power_observed))

