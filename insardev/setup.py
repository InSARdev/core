#!/usr/bin/env python
# ----------------------------------------------------------------------------
# insardev
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev directory for license terms.
# Professional use requires an active per-seat subscription at: https://patreon.com/pechnikov
# ----------------------------------------------------------------------------
from setuptools import setup
import urllib.request

def get_version():
    with open("insardev/__init__.py", "r") as f:
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
    name='insardev',
    version=get_version(),
    description='InSAR.dev (Python InSAR): Satellite Interferometry Framework',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/AlexeyPechnikov/pygmtsar',
    author='Alexey Pechnikov',
    author_email='alexey@pechnikov.dev',
    license='InSAR.dev Personal License',
    license_files=['LICENSE'],
    packages=['insardev'],
    include_package_data=True,
    install_requires=['insardev_toolkit',
                      'xarray',
                      'numpy',
                      'torch',
                      'torch_dct',
                      'ortools',
                      'pandas',
                      'geopandas',
                      'shapely',
                      'rasterio',
                      'rioxarray',
                      'distributed',
                      'dask[complete]',
                      'zarr',
                      'numcodecs',
                      'numba',
                      'pyproj',
                      'scipy',
                      'scikit-learn',
                      'statsmodels>=0.14.0',
                      'matplotlib',
                      'adjustText',
                      'fsspec',
                      'tqdm',
                      'joblib'
                      ],
    extras_require={
                      'vtk_support': ['vtk', 'panel'],
                      'cv': ['opencv-python']
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'License :: Other/Proprietary License',
        'Operating System :: POSIX',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13'
    ],
    python_requires='>=3.11',
    keywords='satellite interferometry, InSAR, remote sensing, geospatial analysis, elevation, Sentinel-1, SBAS, PSI'
)
