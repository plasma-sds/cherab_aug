
import os

from cherab.tools.observers.bolometry import load_bolometer_camera


_JSON_DATA_PATH = os.path.split(__file__)[0]

_CAMERA_MAP = {
    'FDC': 'FDC_camera.json',
    'FHC': 'FHC_camera.json',
    'FHS': 'FHS_camera.json',
    'FLH': 'FLH_camera.json',
    'FLX': 'FLX_camera.json',
    'FVC': 'FVC_camera.json',
}


def load_default_bolometer_config(bolometer_id, parent=None):

    try:
        config_filename = _CAMERA_MAP[bolometer_id]

    except KeyError:
        raise ValueError("Bolometer camera ID '{}' not recognised.".format(bolometer_id))

    return load_bolometer_camera(os.path.join(_JSON_DATA_PATH, config_filename), parent=parent)
