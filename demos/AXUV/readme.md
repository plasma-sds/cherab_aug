# AXUV synthetic diagnostics
Voxel based approach, based on the two bolometry examples in the Cherab documentation:
- [Calculating a Geometry Matrix using the Voxel Framework](https://www.cherab.info/demonstrations/bolometry/geometry_matrix_from_voxels.html)
- [Performing Inversions of Bolometer Measurements Using the Voxel Framework](https://www.cherab.info/demonstrations/bolometry/inversion_with_voxels.html)

Also see a related [issue #469 in the core repository](https://github.com/cherab/core/issues/469).

#### This directory contains two main files:
- `voxels_based_emission.py` - The main script for generating voxel based emission data from 2D plasma parameters during a D/Ne SPI on AUG.
- `emission_to_measurement.ipynb` - Post-processing the emission data with the help of AXUV diode spectral response functions and the generated voxel data, so that we end up with actual synthetic measurements on the diodes.

#### Additionally there are some data files in `/data`:
- `ne.csv` contains the neon lines
- `spi_input.h5` contains an example data input

#### Python library requirements:
- h5py
- shapely
