# InSAR.dev PyGMTSAR

Sentinel-1 SLC preprocessing library for Python.

## Features

- **Per-burst Sentinel-1 TOPS SLC preprocessing** — independent processing of each burst, no frame stitching
- **Geometric coregistration** — bursts aligned to a reference via radar-to-geographic transforms with differential topo phase correction between reference and repeat geometries
- **Dual-polarization support** — all polarization channels (VV+VH or HH+HV) processed together
- **Precise orbit integration** — automatic application of restituted and precise orbit files
- **DEM preparation** — geocoding and coordinate transformations to a common geographic grid
- **User-defined coordinate system and resolution** — output in any EPSG projection at any resolution, from small large-area overviews to precise local analysis grids
- **Solid Earth tidal phase correction** — per-date tidal displacement projected onto the radar line-of-sight using IERS 2003 solid Earth tide model and satellite look vectors, computed efficiently via 2×2 radar-grid corner interpolation
- **Geocoded Zarr output** — burst stacks stored as Zarr v3 with per-pixel azi, rng, and elevation

## License

This software is released under the **BSD 3-Clause License**.
See [LICENSE](./LICENSE) for full terms.

## Contact

- Author: Aleksei Pechnikov
- Email: alexey@pechnikov.dev
- ORCID: https://orcid.org/0000-0001-9626-8615

## Bug Reports

Bug reports and suggestions are welcome via the project's issue tracker.
