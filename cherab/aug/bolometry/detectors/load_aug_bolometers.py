
import os

from cherab.tools.observers.bolometry import load_bolometer_camera


_DATA_PATH = os.path.split(__file__)[0]

_CAMERA_MAP = {
    'FDC': 'FDC_camera.pickle',
    'FHC': 'FHC_camera.pickle',
    'FHS': 'FHS_camera.pickle',
    'FLH': 'FLH_camera.pickle',
    'FLX': 'FLX_camera.pickle',
    'FVC': 'FVC_camera.pickle',
}


def load_default_bolometer_config(bolometer_id, parent=None, inversion_grid=None):

    try:
        config_filename = _CAMERA_MAP[bolometer_id]

    except KeyError:
        raise ValueError("Bolometer camera ID '{}' not recognised.".format(bolometer_id))

    return load_bolometer_camera(os.path.join(_DATA_PATH, config_filename),
                                 parent=parent, inversion_grid=inversion_grid)
