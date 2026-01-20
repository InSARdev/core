[![View on GitHub](https://img.shields.io/badge/GitHub-View%20on%20GitHub-blue)](https://github.com/AlexeyPechnikov/insardev)
[![Support on Patreon](https://img.shields.io/badge/Patreon-Support-orange.svg)](https://www.patreon.com/pechnikov)

## InSAR.dev—Python Ecosystem for Interferometric Synthetic Aperture Radar

## Components

| Package | Description | License |
|---------|-------------|---------|
| [insardev](./insardev/) | Core interferometric processing and analysis | InSARdev-SAL-1.0 |
| [insardev_pygmtsar](./insardev_pygmtsar/) | GMTSAR-based Sentinel-1 SLC preprocessing | BSD 3-Clause |
| [insardev_toolkit](./insardev_toolkit/) | Utility functions and helper tools | BSD 3-Clause |

## Features

- Sentinel-1 SLC bursts processing pipeline
- Interferogram generation, filtering, detrending, unwrapping
- Time series analysis (SBAS, PSI)

## Examples

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pz4QkvJmqagVuLmuLMlfGOSrZLFnJWBw?usp=sharing) **Central Türkiye Earthquakes (2023).** Interferogram covering two consecutive Sentinel-1 scenes (56 bursts) with GPU-accelerated phase unwrapping.

<img src="assets/Türkiye_Earthquakes_2023_intf.jpg" />

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YnXVOpEW8lOkCQ-SV6ki5FN4-YEfKWUL?usp=sharing) **Imperial Valley Subsidence, CA USA (2015).**  This example is provided in the [GMTSAR project](https://topex.ucsd.edu/gmtsar/downloads/) in the archive file [S1A_Stack_CPGF_T173.tar.gz](http://topex.ucsd.edu/gmtsar/tar/S1A_Stack_CPGF_T173.tar.gz), titled 'Sentinel-1 TOPS Time Series'.

The resulting InSAR velocity map is available as a self-contained web page at: [Imperial_Valley_2015.html](https://insar.dev/ui/Imperial_Valley_2015_ipyleaflet.html)

<img src="assets/Imperial_Valley_Subsidence.jpg" />


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KJgX69LyKLfGUkiRLiStR-SMnJOy-eDs?usp=sharing) **Golden Valley Subsidence, CA USA (2021).** This example demonstrates the case study 'Antelope Valley Freeway in Santa Clarita, CA,' as detailed in [SAR Technical Series Part 4 Sentinel-1 global velocity layer: Using global InSAR at scale](https://blog.descarteslabs.com/using-global-insar-at-scale) and [Sentinel-1 Technical Series Part 5 Targeted Analysis](https://blog.descarteslabs.com/sentinel-1-targeted-analysis) with a significant subsidence rate 'exceeding 5cm/year in places'.

<img src="assets/Golden_Valley_Subsidence.jpg" />

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12oss994Pabq_8JDv4Dj-R6iqfGEalHCe?usp=sharing) **Erzincan Elevation, Türkiye (2019).** This example reproduces 29-page ESA document [DEM generation with Sentinel-1 IW](https://step.esa.int/docs/tutorials/S1TBX%20DEM%20generation%20with%20Sentinel-1%20IW%20Tutorial.pdf).

<img src="assets/Türkiye_Elevation_2019_ele.jpg" />

## License

This repository contains components with different licenses:

- **insardev/** - InSAR.dev Source-Available License (see [insardev/LICENSE](./insardev/LICENSE))
- **insardev_pygmtsar/** - BSD 3-Clause License (see [insardev_pygmtsar/LICENSE](./insardev_pygmtsar/LICENSE))
- **insardev_toolkit/** - BSD 3-Clause License (see [insardev_toolkit/LICENSE](./insardev_toolkit/LICENSE))

For Commercial Use of the insardev package, see [insardev/SUBSCRIBE](./insardev/SUBSCRIBE).

## Contact

- Author: Aleksei Pechnikov
- Email: alexey@pechnikov.dev
- ORCID: https://orcid.org/0000-0001-9626-8615

## Bug Reports

Bug reports and suggestions are welcome via the project's issue tracker.
