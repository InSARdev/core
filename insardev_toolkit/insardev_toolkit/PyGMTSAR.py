# ----------------------------------------------------------------------------
# insardev_toolkit
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2026, Alexey Pechnikov
#
# See the LICENSE file in the insardev_toolkit directory for license terms.
# ----------------------------------------------------------------------------
"""
Translate local Sentinel-1 SAFE archives (full SLC scenes or PyGMTSAR fake-SAFE
per-burst downloads) into the InSARdev per-burst directory layout.

PyGMTSAR's burst download writes one or more bursts into a SAFE-shaped tree:

    SAFEDIR/
    └── S1A_IW_SLC__1SDV_..._{hash}.SAFE/
        ├── measurement/{prefix}-{start_lc}-{stop_lc}-{orbit}-{datatake_lc}-{N}.tiff
        ├── annotation/ {prefix}-...xml
        └── annotation/calibration/
            ├── calibration-{prefix}-...xml
            └── noise-{prefix}-...xml

The same tree shape covers full ESA SLC scenes — a subswath TIFF may hold one
burst (PyGMTSAR's old single-burst extract) or many (full scene, ~9 bursts).

InSARdev expects native ASF burst delivery layout:

    DATADIR/
    └── {path:03d}_{burstId}_{IW}/
        ├── measurement/  S1_{burstId}_{IW}_{datetime}_{pol}_{hash}-BURST.nc
        ├── annotation/   S1_..._BURST.xml
        ├── calibration/  S1_..._BURST.xml
        └── noise/        S1_..._BURST.xml

``PyGMTSAR().safeload(SAFEDIR, DATADIR, BURSTS)`` mimics
``ASF.download(DATADIR, BURSTS)`` but reads from local SAFEs instead of the
network. It auto-detects whether each subswath TIFF holds 1 or N bursts and
either renames/copies (1) or extracts a strip + filters XMLs (N).
"""
import os
import shutil
from glob import glob


class PyGMTSAR:
    """Translate local SAFE archives into the InSARdev per-burst layout.

    Mirrors :py:meth:`insardev_toolkit.ASF.download` so callers can swap an
    online download for a local layout-translation step:

        >>> ASF().download(datadir, bursts)             # online
        >>> PyGMTSAR().safeload(safedir, datadir, bursts)   # offline

    Handles both single-burst-per-SAFE (PyGMTSAR fake SAFE) and full-subswath
    SAFE (ESA SLC scenes downloaded directly). For multi-burst TIFFs, each
    requested burst's pixel strip is sliced out and written as a standalone
    TIFF; the subswath annotation/noise/calibration XMLs are filtered to the
    single burst's entries with line numbers re-zeroed.
    """

    def safeload(self, safedir, datadir, bursts, skip_errors=False, skip_exist=True, debug=False):
        """Translate SAFE-format bursts in ``safedir`` into InSARdev layout under ``datadir``.

        A one-time translation: the measurement is converted into the burst's compressed NetCDF4 file, and the
        XMLs and orbits, small next to it, are copied.

        Parameters
        ----------
        safedir : str
            Source directory containing ``*.SAFE/`` archives and (optionally)
            orbit ``*.EOF`` files at its root.
        datadir : str
            Target directory for InSARdev per-burst layout. Created if missing.
        bursts : str, list, or geopandas.GeoDataFrame
            ASF-format burst identifiers, e.g.
            ``'S1_262885_IW2_20190702T032452_VV_69C5-BURST'``. Accepts a single
            string, newline-separated string, list of strings, or a GeoDataFrame
            with a ``sceneName`` column (as ``asf_search`` returns).
        skip_errors : bool, optional
            If False (default), raise on missing/ambiguous source files.
            If True, log a warning and continue with the next burst.
        skip_exist : bool, optional
            Skip bursts already present in ``datadir`` (all four files exist
            and are non-empty). Default True.
        debug : bool, optional
            Print per-burst progress. Default False.

        Returns
        -------
        pandas.DataFrame
            One row per burst attempted with columns
            ``[burst, status, path, burstId, subswath, target_dir]``.
            ``status`` is ``'created'``, ``'skipped'``, or ``'error: <reason>'``.

        Examples
        --------
        >>> from insardev_toolkit import PyGMTSAR
        >>> bursts = [
        ...     'S1_262885_IW2_20190702T032452_VV_69C5-BURST',
        ...     'S1_262886_IW2_20190702T032455_VV_69C5-BURST',
        ... ]
        >>> PyGMTSAR().safeload('pygmtsar_data/', 'insardev_data/', bursts)
        """
        import pandas as pd

        try:
            import geopandas as gpd
            is_gdf = isinstance(bursts, gpd.GeoDataFrame)
        except ImportError:
            is_gdf = False
        if is_gdf:
            bursts = bursts['sceneName'].tolist()
        elif isinstance(bursts, str):
            bursts = list(filter(None, map(str.strip, bursts.split('\n'))))

        os.makedirs(datadir, exist_ok=True)

        safe_index = self._index_safes(safedir)
        if debug:
            print(f'safeload: found {len(safe_index)} SAFE dir(s) in {safedir}')

        records = []
        for burst in bursts:
            try:
                rec = self._process_burst(burst, datadir, safe_index, skip_exist=skip_exist, debug=debug)
            except Exception as e:
                if not skip_errors:
                    raise
                rec = {'burst': burst, 'status': f'error: {e}', 'path': None,
                       'burstId': None, 'subswath': None, 'target_dir': None}
                print(f'WARNING: {burst}: {e}')
            records.append(rec)

        self._transfer_orbits(safedir, datadir, debug=debug)

        return pd.DataFrame.from_records(records)

    # ---- SAFE indexing ----

    @staticmethod
    def _index_safes(safedir):
        """Map 4-char scene hash → list of SAFE paths. The same hash can appear
        in multiple SAFEs in pathological cases (e.g. duplicate downloads);
        we keep all and let the matcher fail loudly if ambiguous."""
        index = {}
        for safe in glob(os.path.join(safedir, '*.SAFE')):
            name = os.path.basename(safe)
            stem = name[:-5] if name.endswith('.SAFE') else name
            parts = stem.split('_')
            if len(parts) < 10:
                continue
            scene_hash = parts[9].upper()
            index.setdefault(scene_hash, []).append(safe)
        return index

    # ---- main per-burst worker ----

    @classmethod
    def _process_burst(cls, burst, datadir, safe_index, skip_exist, debug):
        import xmltodict

        burst_id_num, subswath, datetime_str, pol, scene_hash = cls._parse_burst_id(burst)

        safes = safe_index.get(scene_hash, [])
        if not safes:
            raise FileNotFoundError(
                f'no SAFE dir with hash {scene_hash} for {burst}'
            )
        if len(safes) == 1:
            safe = safes[0]
        else:
            # Modern PyGMTSAR creates one SAFE per burst sharing the same scene
            # hash; disambiguate by matching the SAFE name's start_time (parts[5])
            # against the burst's datetime.
            matches = []
            for s in safes:
                parts = os.path.basename(s)[:-len('.SAFE')].split('_')
                if len(parts) >= 6 and parts[5] == datetime_str:
                    matches.append(s)
            if not matches:
                # SAFEs may bracket the burst time; pick the SAFE whose start <=
                # burst datetime <= stop. Fall back to reading annotations if
                # name-based matching fails.
                safe = cls._pick_safe_by_annotation(safes, datetime_str, subswath, pol, burst)
            elif len(matches) > 1:
                raise ValueError(
                    f'ambiguous SAFE match for {burst}: {matches}'
                )
            else:
                safe = matches[0]

        # Locate the (subswath, polarisation) source files inside the SAFE.
        mission = os.path.basename(safe).split('_')[0]    # 'S1A'
        prefix_lc = f'{mission.lower()}-{subswath.lower()}-slc-{pol.lower()}'
        src_meas, src_ann, src_cal, src_noise = cls._find_safe_files(safe, prefix_lc)

        # Parse annotation, find the burstIndex for the requested datetime.
        with open(src_ann, 'r') as f:
            ann_xml = f.read()
        annotation = xmltodict.parse(ann_xml)['product']
        burst_index, lines_per_burst, samples_per_burst = cls._find_burst_index(
            annotation, datetime_str, burst,
        )

        # pathNumber from absoluteOrbitNumber + platform-specific cycle offset.
        # (S1A starts cycle at abs_orbit 74; S1B at 28; S1C at 173.)
        abs_orbit = int(annotation['adsHeader']['absoluteOrbitNumber'])
        platform = annotation['adsHeader']['missionId']  # 'S1A' / 'S1B' / 'S1C'
        offset = {'S1A': 73, 'S1B': 27, 'S1C': 172}.get(platform)
        if offset is None:
            raise ValueError(f"unknown platform {platform!r} in {burst}: cannot compute pathNumber")
        path = ((abs_orbit - offset) % 175) + 1

        # Plan target paths. The burst is stored as <burst>.nc; bursts translated before keep their <burst>.tiff.
        from .utils_S1 import measurement_path
        target_dir = os.path.join(datadir, f'{path:03d}_{burst_id_num}_{subswath}')
        tgt_meas  = os.path.join(target_dir, 'measurement', f'{burst}.nc')
        tgt_ann   = os.path.join(target_dir, 'annotation',  f'{burst}.xml')
        tgt_cal   = os.path.join(target_dir, 'calibration', f'{burst}.xml')
        tgt_noise = os.path.join(target_dir, 'noise',       f'{burst}.xml')
        present_meas = measurement_path(os.path.join(target_dir, 'measurement'), burst)

        if skip_exist and all(cls._is_present(p) for p in (present_meas, tgt_ann, tgt_cal, tgt_noise)):
            if debug:
                print(f'  skip {burst} (already present)')
            return {'burst': burst, 'status': 'skipped', 'path': path,
                    'burstId': int(burst_id_num), 'subswath': subswath,
                    'target_dir': target_dir}

        for sub in ('measurement', 'annotation', 'calibration', 'noise'):
            os.makedirs(os.path.join(target_dir, sub), exist_ok=True)

        # Count bursts in the source annotation. 1 burst → rename/copy whole TIFF.
        # >1 → real burst extraction.
        burst_list = annotation['swathTiming']['burstList']['burst']
        if not isinstance(burst_list, list):
            burst_list = [burst_list]
        n_bursts_in_safe = len(burst_list)

        if n_bursts_in_safe == 1:
            cls._transfer_singleburst(src_meas, src_ann, src_cal, src_noise,
                                      tgt_meas, tgt_ann, tgt_cal, tgt_noise)
            mode = 'converted'
        else:
            cls._extract_multiburst(
                src_meas, src_cal, src_noise,
                tgt_meas, tgt_ann, tgt_cal, tgt_noise,
                annotation, burst_index, lines_per_burst, samples_per_burst,
            )
            mode = f'extracted (burst {burst_index + 1}/{n_bursts_in_safe})'

        if debug:
            print(f'  {mode} {burst} -> {target_dir}')
        return {'burst': burst, 'status': 'created', 'path': path,
                'burstId': int(burst_id_num), 'subswath': subswath,
                'target_dir': target_dir}

    # ---- single-burst SAFE: convert the measurement, copy the XMLs ----

    @classmethod
    def _transfer_singleburst(cls, src_meas, src_ann, src_cal, src_noise,
                              tgt_meas, tgt_ann, tgt_cal, tgt_noise):
        from .utils_S1 import write_slc
        with open(src_meas, 'rb') as f:
            write_slc(f.read(), tgt_meas)
        for src, tgt in (
            (src_ann, tgt_ann),
            (src_cal, tgt_cal),
            (src_noise, tgt_noise),
        ):
            cls._transfer(src, tgt)

    # ---- multi-burst SAFE: extract TIFF strip + filter XMLs ----

    @classmethod
    def _extract_multiburst(cls, src_meas, src_cal, src_noise,
                            tgt_meas, tgt_ann, tgt_cal, tgt_noise,
                            annotation, burst_index, lines_per_burst, samples_per_burst):
        import xmltodict

        # 1. Read the requested burst's pixel strip from the multi-burst TIFF
        #    as a standalone single-burst TIFF, in memory.
        import io
        from .utils_S1 import write_slc
        tiff_bytes = cls._burst_tiff_bytes(src_meas, burst_index, lines_per_burst)

        # 2. The new TIFF's first-strip byte offset for the annotation XML.
        from tifffile import TiffFile
        with TiffFile(io.BytesIO(tiff_bytes)) as tif:
            page = tif.pages[0]
            actual_lines, actual_samples = page.shape
            tiff_offset = int(page.dataoffsets[0])
        if actual_lines != lines_per_burst or actual_samples != samples_per_burst:
            raise RuntimeError(
                f'TIFF dimension mismatch after extraction: got '
                f'{actual_lines}x{actual_samples}, expected '
                f'{lines_per_burst}x{samples_per_burst}'
            )
        # the burst is stored as NetCDF4, which keeps that offset too
        write_slc(tiff_bytes, tgt_meas)

        # 3. Filter annotation/noise/calibration XMLs to the single requested burst.
        ann_xml_out = cls._build_product_xml(annotation, burst_index, tiff_offset)
        cls._atomic_write_text(tgt_ann, ann_xml_out)

        # Re-parse the new annotation to recover start_utc / stop_utc.
        new_ann = xmltodict.parse(ann_xml_out)['product']
        start_utc = new_ann['adsHeader']['startTime']
        stop_utc = new_ann['adsHeader']['stopTime']

        with open(src_noise, 'r') as f:
            src_noise_dict = xmltodict.parse(f.read())['noise']
        noise_xml_out = cls._build_noise_xml(
            src_noise_dict, burst_index, lines_per_burst, start_utc, stop_utc,
        )
        cls._atomic_write_text(tgt_noise, noise_xml_out)

        with open(src_cal, 'r') as f:
            src_cal_dict = xmltodict.parse(f.read())['calibration']
        cal_xml_out = cls._build_calibration_xml(
            src_cal_dict, burst_index, lines_per_burst, start_utc, stop_utc,
        )
        cls._atomic_write_text(tgt_cal, cal_xml_out)

    # ---- per-burst XML builders (lifted from ASF/CDSE download paths) ----

    @staticmethod
    def _filter_azimuth_time(items, start_utc_dt, stop_utc_dt, delta=3):
        """Keep XML entries with azimuthTime in [start - delta, stop + delta] sec."""
        from datetime import datetime, timedelta
        if not isinstance(items, list):
            items = [items]
        return [
            item for item in items
            if datetime.strptime(item['azimuthTime'], '%Y-%m-%dT%H:%M:%S.%f')
                >= start_utc_dt - timedelta(seconds=delta)
            and datetime.strptime(item['azimuthTime'], '%Y-%m-%dT%H:%M:%S.%f')
                <= stop_utc_dt + timedelta(seconds=delta)
        ]

    @classmethod
    def _build_product_xml(cls, annotation, burst_index, tiff_offset):
        """Build a single-burst product/annotation XML from a multi-burst subswath product dict."""
        from datetime import datetime, timedelta
        import xmltodict

        lines_per_burst = int(annotation['swathTiming']['linesPerBurst'])
        burst_list = annotation['swathTiming']['burstList']['burst']
        if not isinstance(burst_list, list):
            burst_list = [burst_list]
        start_utc = burst_list[burst_index]['azimuthTime']
        start_utc_dt = datetime.strptime(start_utc, '%Y-%m-%dT%H:%M:%S.%f')
        azimuth_time_interval = float(
            annotation['imageAnnotation']['imageInformation']['azimuthTimeInterval']
        )
        stop_utc_dt = start_utc_dt + timedelta(
            seconds=(lines_per_burst - 1) * azimuth_time_interval
        )
        stop_utc = stop_utc_dt.strftime('%Y-%m-%dT%H:%M:%S.%f')

        product = {}

        adsHeader = annotation['adsHeader']
        adsHeader['startTime'] = start_utc
        adsHeader['stopTime'] = stop_utc
        adsHeader['imageNumber'] = '001'
        product['adsHeader'] = adsHeader

        if 'qualityInformation' in annotation:
            qi = {}
            if 'productQualityIndex' in annotation['qualityInformation']:
                qi['productQualityIndex'] = annotation['qualityInformation']['productQualityIndex']
            if 'qualityDataList' in annotation['qualityInformation']:
                qi['qualityDataList'] = annotation['qualityInformation']['qualityDataList']
            product['qualityInformation'] = qi

        if 'generalAnnotation' in annotation:
            product['generalAnnotation'] = annotation['generalAnnotation']

        imageAnnotation = annotation['imageAnnotation']
        imageAnnotation['imageInformation']['productFirstLineUtcTime'] = start_utc
        imageAnnotation['imageInformation']['productLastLineUtcTime'] = stop_utc
        imageAnnotation['imageInformation']['productComposition'] = 'Assembled'
        imageAnnotation['imageInformation']['sliceNumber'] = '0'
        imageAnnotation['imageInformation']['sliceList'] = {'@count': '0'}
        imageAnnotation['imageInformation']['numberOfLines'] = str(lines_per_burst)
        product['imageAnnotation'] = imageAnnotation

        if 'dopplerCentroid' in annotation:
            dop = annotation['dopplerCentroid']
            items = cls._filter_azimuth_time(dop['dcEstimateList']['dcEstimate'], start_utc_dt, stop_utc_dt)
            dop['dcEstimateList'] = {'@count': len(items), 'dcEstimate': items}
            product['dopplerCentroid'] = dop

        if 'antennaPattern' in annotation:
            ant = annotation['antennaPattern']
            items = cls._filter_azimuth_time(ant['antennaPatternList']['antennaPattern'], start_utc_dt, stop_utc_dt)
            ant['antennaPatternList'] = {'@count': len(items), 'antennaPattern': items}
            product['antennaPattern'] = ant

        swathTiming = annotation['swathTiming']
        items = cls._filter_azimuth_time(swathTiming['burstList']['burst'], start_utc_dt, start_utc_dt, 1)
        assert len(items) == 1, f'expected 1 burst after azimuth-time filter, got {len(items)}'
        items[0]['byteOffset'] = tiff_offset
        swathTiming['burstList'] = {'@count': len(items), 'burst': items}
        product['swathTiming'] = swathTiming

        geolocationGrid = annotation['geolocationGrid']
        geoloc_points = geolocationGrid['geolocationGridPointList']['geolocationGridPoint']
        if not isinstance(geoloc_points, list):
            geoloc_points = [geoloc_points]
        items = cls._filter_azimuth_time(geoloc_points, start_utc_dt, stop_utc_dt, 1)
        for item in items:
            item['line'] = str(int(item['line']) - (lines_per_burst * burst_index))
        geolocationGrid['geolocationGridPointList'] = {
            '@count': len(items), 'geolocationGridPoint': items
        }
        product['geolocationGrid'] = geolocationGrid

        if 'coordinateConversion' in annotation:
            product['coordinateConversion'] = annotation['coordinateConversion']
        if 'swathMerging' in annotation:
            product['swathMerging'] = annotation['swathMerging']

        return xmltodict.unparse({'product': product}, pretty=True, indent='  ')

    @classmethod
    def _build_noise_xml(cls, noise_annotation, burst_index, lines_per_burst, start_utc, stop_utc):
        """Build a single-burst noise XML from the subswath noise dict."""
        from datetime import datetime
        import xmltodict

        start_utc_dt = datetime.strptime(start_utc, '%Y-%m-%dT%H:%M:%S.%f')
        stop_utc_dt = datetime.strptime(stop_utc, '%Y-%m-%dT%H:%M:%S.%f')

        noise = {}

        if 'adsHeader' in noise_annotation:
            adsHeader = noise_annotation['adsHeader']
            adsHeader['startTime'] = start_utc
            adsHeader['stopTime'] = stop_utc
            adsHeader['imageNumber'] = '001'
            noise['adsHeader'] = adsHeader

        if 'noiseVectorList' in noise_annotation:
            nv_list = noise_annotation['noiseVectorList']
            nv_items = nv_list.get('noiseVector', [])
            if not isinstance(nv_items, list):
                nv_items = [nv_items]
            items = cls._filter_azimuth_time(nv_items, start_utc_dt, stop_utc_dt)
            for item in items:
                item['line'] = str(int(item['line']) - (lines_per_burst * burst_index))
            noise['noiseVectorList'] = {'@count': len(items), 'noiseVector': items}

        if 'noiseRangeVectorList' in noise_annotation:
            nrv_list = noise_annotation['noiseRangeVectorList']
            nrv_items = nrv_list.get('noiseRangeVector', [])
            if not isinstance(nrv_items, list):
                nrv_items = [nrv_items]
            items = cls._filter_azimuth_time(nrv_items, start_utc_dt, stop_utc_dt)
            for item in items:
                item['line'] = str(int(item['line']) - (lines_per_burst * burst_index))
            noise['noiseRangeVectorList'] = {'@count': len(items), 'noiseRangeVector': items}

        if 'noiseAzimuthVectorList' in noise_annotation:
            nav_list = noise_annotation['noiseAzimuthVectorList']
            nav = nav_list.get('noiseAzimuthVector', {})
            if nav and 'line' in nav:
                line_data = nav['line']
                if isinstance(line_data, dict) and '#text' in line_data:
                    line_items = [int(x) for x in line_data['#text'].split()]
                elif isinstance(line_data, str):
                    line_items = [int(x) for x in line_data.split()]
                else:
                    line_items = []

                if line_items:
                    lowers = [item for item in line_items if item <= burst_index * lines_per_burst] or [line_items[0]]
                    uppers = [item for item in line_items if item >= (burst_index + 1) * lines_per_burst - 1] or [line_items[-1]]
                    mask = [lowers[-1] <= item <= uppers[0] for item in line_items]
                    filtered_lines = [item - burst_index * lines_per_burst for item, m in zip(line_items, mask) if m]

                    nav['firstAzimuthLine'] = str(lowers[-1] - burst_index * lines_per_burst)
                    nav['lastAzimuthLine'] = str(uppers[0] - burst_index * lines_per_burst)
                    nav['line'] = {'@count': str(len(filtered_lines)),
                                   '#text': ' '.join(str(x) for x in filtered_lines)}

                    if 'noiseAzimuthLut' in nav:
                        lut_data = nav['noiseAzimuthLut']
                        if isinstance(lut_data, dict) and '#text' in lut_data:
                            lut_items = lut_data['#text'].split()
                        elif isinstance(lut_data, str):
                            lut_items = lut_data.split()
                        else:
                            lut_items = []
                        filtered_lut = [item for item, m in zip(lut_items, mask) if m]
                        nav['noiseAzimuthLut'] = {'@count': str(len(filtered_lut)),
                                                  '#text': ' '.join(filtered_lut)}

                    noise['noiseAzimuthVectorList'] = {'noiseAzimuthVector': nav}

        return xmltodict.unparse({'noise': noise}, pretty=True, indent='  ')

    @classmethod
    def _build_calibration_xml(cls, calibration_annotation, burst_index, lines_per_burst, start_utc, stop_utc):
        """Build a single-burst calibration XML from the subswath calibration dict."""
        from datetime import datetime
        import xmltodict

        start_utc_dt = datetime.strptime(start_utc, '%Y-%m-%dT%H:%M:%S.%f')
        stop_utc_dt = datetime.strptime(stop_utc, '%Y-%m-%dT%H:%M:%S.%f')

        calibration = {}

        if 'adsHeader' in calibration_annotation:
            adsHeader = calibration_annotation['adsHeader']
            adsHeader['startTime'] = start_utc
            adsHeader['stopTime'] = stop_utc
            adsHeader['imageNumber'] = '001'
            calibration['adsHeader'] = adsHeader

        if 'calibrationInformation' in calibration_annotation:
            calibration['calibrationInformation'] = calibration_annotation['calibrationInformation']

        if 'calibrationVectorList' in calibration_annotation:
            cv_list = calibration_annotation['calibrationVectorList']
            cv_items = cv_list.get('calibrationVector', [])
            if not isinstance(cv_items, list):
                cv_items = [cv_items]
            items = cls._filter_azimuth_time(cv_items, start_utc_dt, stop_utc_dt)
            for item in items:
                item['line'] = str(int(item['line']) - (lines_per_burst * burst_index))
            calibration['calibrationVectorList'] = {'@count': len(items), 'calibrationVector': items}

        return xmltodict.unparse({'calibration': calibration}, pretty=True, indent='  ')

    @staticmethod
    def _burst_tiff_bytes(src_tiff, burst_index, lines_per_burst):
        """Extract one burst's rows from a multi-burst Sentinel-1 SLC TIFF as a standalone single-burst
        complex int16 TIFF held in memory.

        A Sentinel-1 SLC TIFF is uncompressed with contiguous strips, so the burst's rows are read straight from
        the file, without decoding the whole subswath; any other layout is decoded by tifffile."""
        import numpy as np
        from tifffile import TiffFile
        from .utils_S1 import pairs_tiff

        first, last = burst_index * lines_per_burst, (burst_index + 1) * lines_per_burst
        with TiffFile(src_tiff) as tif:
            page = tif.pages[0]
            if page.shape[0] < last:
                raise ValueError(
                    f'TIFF too short for burst_index={burst_index}: '
                    f'page has {page.shape[0]} rows, need at least {last}'
                )
            lines, samples = (int(n) for n in page.shape)
            offsets, counts = page.dataoffsets, page.databytecounts
            contiguous = (page.dtype == np.complex64 and int(page.compression) == 1 and len(offsets) == lines
                          and sum(counts) == lines * samples * 4
                          and all(offsets[i] + counts[i] == offsets[i + 1] for i in range(len(offsets) - 1)))
            if contiguous:
                with open(src_tiff, 'rb') as f:
                    f.seek(int(offsets[first]))
                    raw = f.read(lines_per_burst * samples * 4)
                iq = np.frombuffer(raw, dtype=tif.byteorder + 'i2').reshape(lines_per_burst, samples, 2)
            else:
                c = page.asarray()[first:last]
                iq = c.view(np.float32).reshape(c.shape + (2,)).astype(np.int16)
        return pairs_tiff(iq)

    # ---- helpers ----

    @staticmethod
    def _parse_burst_id(burst):
        if not (burst.startswith('S1_') and burst.endswith('-BURST')):
            raise ValueError(f"not an ASF S1 burst ID: '{burst}'")
        body = burst[3:-len('-BURST')]
        parts = body.split('_')
        if len(parts) != 5:
            raise ValueError(
                f"malformed burst ID '{burst}': expected 5 underscore-separated fields after S1_"
            )
        burst_id_num, subswath, datetime_str, pol, scene_hash = parts
        return burst_id_num, subswath, datetime_str, pol, scene_hash.upper()

    @staticmethod
    def _find_safe_files(safe, prefix_lc):
        """Locate the (annotation, noise, calibration, measurement) files for a
        given (subswath, polarisation) prefix inside the SAFE."""
        meas = glob(os.path.join(safe, 'measurement', f'{prefix_lc}-*.tiff'))
        ann  = glob(os.path.join(safe, 'annotation',  f'{prefix_lc}-*.xml'))
        cal  = glob(os.path.join(safe, 'annotation', 'calibration', f'calibration-{prefix_lc}-*.xml'))
        noi  = glob(os.path.join(safe, 'annotation', 'calibration', f'noise-{prefix_lc}-*.xml'))
        for label, found in (('measurement TIFF', meas), ('annotation XML', ann),
                             ('calibration XML', cal), ('noise XML', noi)):
            if not found:
                raise FileNotFoundError(f'{label} for {prefix_lc} not found in {safe}')
            if len(found) > 1:
                raise ValueError(f'multiple {label} files match {prefix_lc} in {safe}: {found}')
        return meas[0], ann[0], cal[0], noi[0]

    @classmethod
    def _pick_safe_by_annotation(cls, safes, datetime_str, subswath, pol, burst):
        """Fallback: open each candidate SAFE's annotation and pick the one whose
        subswath/polarisation matches and whose burstList contains an azimuthTime
        within 1s of the burst datetime."""
        from datetime import datetime
        import xmltodict
        target = datetime.strptime(datetime_str, '%Y%m%dT%H%M%S')
        for s in safes:
            mission = os.path.basename(s).split('_')[0]
            prefix_lc = f'{mission.lower()}-{subswath.lower()}-slc-{pol.lower()}'
            ann_glob = glob(os.path.join(s, 'annotation', f'{prefix_lc}-*.xml'))
            if not ann_glob:
                continue
            with open(ann_glob[0], 'r') as f:
                ann = xmltodict.parse(f.read())['product']
            blist = ann['swathTiming']['burstList']['burst']
            if not isinstance(blist, list):
                blist = [blist]
            for b in blist:
                t = datetime.strptime(b['azimuthTime'], '%Y-%m-%dT%H:%M:%S.%f')
                if abs((t - target).total_seconds()) <= 1.0:
                    return s
        raise FileNotFoundError(
            f'no SAFE among {safes} contains burst {burst} (datetime {datetime_str})'
        )

    @staticmethod
    def _find_burst_index(annotation, datetime_str, burst):
        """Return (burst_index, lines_per_burst, samples_per_burst) for the burst
        whose ``azimuthTime`` matches ``datetime_str`` ('YYYYMMDDTHHMMSS').

        Match is by truncated start_time (second precision) since the ASF burst
        ID encodes only seconds while ``azimuthTime`` has microseconds.
        """
        from datetime import datetime

        lines_per_burst = int(annotation['swathTiming']['linesPerBurst'])
        samples_per_burst = int(
            annotation['imageAnnotation']['imageInformation']['numberOfSamples']
        )

        burst_list = annotation['swathTiming']['burstList']['burst']
        if not isinstance(burst_list, list):
            burst_list = [burst_list]

        target = datetime.strptime(datetime_str, '%Y%m%dT%H%M%S')
        for idx, b in enumerate(burst_list):
            t = datetime.strptime(b['azimuthTime'], '%Y-%m-%dT%H:%M:%S.%f')
            if t.replace(microsecond=0) == target:
                return idx, lines_per_burst, samples_per_burst

        # Fallback: pick the nearest by absolute time (handles minor truncation drift).
        diffs = [abs((datetime.strptime(b['azimuthTime'], '%Y-%m-%dT%H:%M:%S.%f') - target).total_seconds())
                 for b in burst_list]
        idx = min(range(len(diffs)), key=diffs.__getitem__)
        if diffs[idx] > 1.0:
            raise ValueError(
                f'no burst within 1s of {datetime_str} in SAFE annotation for {burst}; '
                f'closest azimuthTime differs by {diffs[idx]:.3f}s'
            )
        return idx, lines_per_burst, samples_per_burst

    @staticmethod
    def _is_present(path):
        return os.path.isfile(path) and os.path.getsize(path) > 0

    @staticmethod
    def _transfer(src, tgt):
        if os.path.lexists(tgt):
            os.remove(tgt)
        shutil.copy2(src, tgt)

    @staticmethod
    def _atomic_write_text(tgt, content):
        tmp = tgt + '.tmp'
        with open(tmp, 'w') as f:
            f.write(content)
        os.replace(tmp, tgt)

    @classmethod
    def _transfer_orbits(cls, safedir, datadir, debug):
        for eof in glob(os.path.join(safedir, '*.EOF')):
            tgt = os.path.join(datadir, os.path.basename(eof))
            if cls._is_present(tgt):
                continue
            cls._transfer(eof, tgt)
            if debug:
                print(f'  orbit copied: {os.path.basename(eof)}')
