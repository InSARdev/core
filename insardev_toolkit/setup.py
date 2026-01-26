#!/usr/bin/env python
# ----------------------------------------------------------------------------
# insardev_toolkit
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_toolkit directory for license terms.
# ----------------------------------------------------------------------------
from setuptools import setup
import urllib.request

def get_version():
    with open("insardev_toolkit/__init__.py", "r") as f:
        for line in f:
            if line.startswith("__version__"):
                version = line.split('=')[1]
                version = version.replace("'", "").replace('"', "").strip()
                return version

# read the contents of local README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

#upstream_url = 'https://raw.githubusercontent.com/AlexeyPechnikov/InSARdev/refs/heads/main/README.md'
#response = urllib.request.urlopen(upstream_url)
#long_description = response.read().decode('utf-8')

setup(
    name='insardev_toolkit',
    version=get_version(),
    description='InSAR.dev (Python InSAR): Geospatial Processing Toolkit',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/AlexeyPechnikov/InSARdev',
    author='Alexey Pechnikov',
    author_email='alexey@pechnikov.dev',
    license='BSD-3-Clause',
    packages=['insardev_toolkit'],
    include_package_data=True,
    install_requires=['xarray>=2024.1.0',
                      'numpy',
                      'pandas>=2.2',
                      'geopandas',
                      'distributed>=2024.1.0',
                      'dask[complete]>=2024.4.1',
                      'zarr',
                      'joblib',
                      'tqdm',
                      'ipywidgets',
                      'shapely>=2.0.2',
                      'xmltodict',
                      'rioxarray',
                      'tifffile',
                      'netCDF4',
                      'nc-time-axis',
                      'remotezip',
                      'asf_search',
                      'matplotlib',
                      'imageio'
                      ],
    extras_require={
                      'vtk_support': ['vtk', 'panel']
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13'
    ],
    python_requires='>=3.11',
    keywords='remote sensing, geospatial analysis, DEM, topography, SRTM, Copernicus, ALOS, OpenStreetMap, OSM, Google Maps, ASF, NetCDF, GeoTIFF, VTK'
)
