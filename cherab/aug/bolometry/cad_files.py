
import os

try:
    CADMESH_PATH = os.environ['CHERAB_CADMESH']
except KeyError:
    raise ValueError("CHERAB's CAD file path environment variable 'CHERAB_CADMESH' is not set.")


FDC_TUBE = [os.path.join(CADMESH_PATH, 'aug/diagnostics/bolometry/FDC_tube.rsm')]

FLH_BOX = [os.path.join(CADMESH_PATH, 'aug/diagnostics/bolometry/FLH_box.rsm')]
FLH_TUBE = [os.path.join(CADMESH_PATH, 'aug/diagnostics/bolometry/FLH_tube.rsm')]
FLH = FLH_BOX + FLH_TUBE

FHS_BOX = [os.path.join(CADMESH_PATH, 'aug/diagnostics/bolometry/FHS_box.rsm')]
FHS_TUBE = [os.path.join(CADMESH_PATH, 'aug/diagnostics/bolometry/FHS_tube.rsm')]
FHS = FHS_BOX + FHS_TUBE

FLX_BOX = [os.path.join(CADMESH_PATH, 'aug/diagnostics/bolometry/FLX_box.rsm')]
FLX_TUBE = [os.path.join(CADMESH_PATH, 'aug/diagnostics/bolometry/FLX_tube.rsm')]
FLX = FLX_BOX + FLX_TUBE

FVC_BOX = [os.path.join(CADMESH_PATH, 'aug/diagnostics/bolometry/FVC_box.rsm')]
FVC_TUBE = [os.path.join(CADMESH_PATH, 'aug/diagnostics/bolometry/FVC_tube.rsm')]
FVC = FVC_BOX + FVC_TUBE
