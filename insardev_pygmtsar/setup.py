#!/usr/bin/env python
# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from setuptools import setup
import urllib.request

def get_version():
    with open("insardev_pygmtsar/__init__.py", "r") as f:
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
    name='insardev_pygmtsar',
    version=get_version(),
    description='InSAR.dev (Python InSAR): PyGMTSAR backend',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/AlexeyPechnikov/InSARdev',
    author='Alexey Pechnikov',
    author_email='alexey@pechnikov.dev',
    license='BSD-3-Clause',
    packages=['insardev_pygmtsar'],
    include_package_data=True,
    package_data={
        'insardev_pygmtsar': ['data/geoid_egm96_icgem.grd'],
    },
    install_requires=['insardev_toolkit',
                      'xarray[complete]',
                      'numpy',
                      'pandas>=2.2',
                      'geopandas',
                      'zarr',
                      'fsspec',
                      'opencv-python',
                      'scipy',
                      'scikit-learn',
                      'shapely>=2.0.2',
                      'xmltodict',
                      'rioxarray',
                      'statsmodels>=0.14.0',
                      'h5netcdf',
                      'h5py',
                      'tifffile',
                      'netCDF4',
                      'pyproj',
                      'joblib',
                      'tqdm',
                      'matplotlib',
                      'python-dateutil',
                      ],
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
    keywords='satellite interferometry, InSAR, remote sensing, Sentinel-1'
)
