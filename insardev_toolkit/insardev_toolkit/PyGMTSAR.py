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
``ASF.download(DATADIR, BURSTS)`` but reads the bursts from local SAFEs: only
their catalog records come from the network, in one ASF catalog search, which
finds each burst and names its directory as ASF.download does. It auto-detects
whether each subswath TIFF holds 1 or N bursts and either converts it (1) or
extracts a strip (N); the XMLs are filtered to the burst as the downloads write
them (utils_S1.burst_xmls), so both give the files of ASF.download.
"""
import os
from glob import glob
from .utils_files import exists, EmptyFileError


class PyGMTSAR:
    """Translate local SAFE archives into the InSARdev per-burst layout.

    Mirrors :py:meth:`insardev_toolkit.ASF.download` so callers can swap an
    online download for a local layout-translation step:

        >>> ASF().download(datadir, bursts)             # online
        >>> PyGMTSAR().safeload(safedir, datadir, bursts)   # local SAFEs, one catalog search

    Handles both single-burst-per-SAFE (PyGMTSAR fake SAFE) and full-subswath
    SAFE (ESA SLC scenes downloaded directly). For multi-burst TIFFs, each
    requested burst's pixel strip is sliced out; the annotation/noise/calibration
    XMLs are filtered to the single burst's entries with line numbers re-zeroed,
    as the downloads write them (utils_S1.burst_xmls).
    """

    def safeload(self, safedir, datadir, bursts, skip_errors=False, skip_exist=True, debug=False):
        """Translate SAFE-format bursts in ``safedir`` into InSARdev layout under ``datadir``.

        A one-time translation: the measurement is converted into the burst's compressed NetCDF4 file, the XMLs
        are written as ASF.download writes them (utils_S1.burst_xmls), and the orbits are copied.

        Every requested burst is looked up by its name in the ASF catalog, in one batched search as
        ASF.download does, so this needs the network. The burst directory is the catalog's fullBurstID,
        the directory ASF.download writes; the files keep the requested name. A name the catalog does not
        return is an error, and so is a SAFE annotation without a burst at the second of the name.

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
            with a ``sceneName`` column (as ``ASF.search`` returns).
        skip_errors : bool, optional
            If False (default), raise on a burst missing from the catalog or on
            missing/ambiguous source files. If True, log a warning and continue
            with the next burst. An empty source or target file always raises.
        skip_exist : bool, optional
            Skip bursts already present in ``datadir`` (all four files exist).
            Default True.
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
        >>> # the SAFEs hold the bursts; one ASF catalog search names the directories 123_262885_IW2, 123_262886_IW2
        >>> PyGMTSAR().safeload('pygmtsar_data/', 'insardev_data/', bursts)
        """
        import pandas as pd
        from .ASF import _asf_burst_records

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

        # find every burst in the ASF catalog, in one batched search, as ASF.download finds the bursts it downloads
        catalog = {record.geojson()['properties']['fileID']: record.geojson()['properties']
                   for record in _asf_burst_records(bursts)}

        safe_index = self._index_safes(safedir)
        if debug:
            print(f'safeload: found {len(safe_index)} SAFE dir(s) in {safedir}')

        # an empty file among the translated bursts and the orbits raises before any burst is translated
        self._check_existing(safedir, datadir, bursts, catalog)

        records = []
        for burst in bursts:
            try:
                rec = self._process_burst(burst, catalog.get(burst), datadir, safe_index,
                                          skip_exist=skip_exist, debug=debug)
            except Exception as e:
                if not skip_errors or isinstance(e, EmptyFileError):
                    raise
                rec = {'burst': burst, 'status': f'error: {e}', 'path': None,
                       'burstId': None, 'subswath': None, 'target_dir': None}
                print(f'WARNING: {burst}: {e}')
            records.append(rec)

        self._transfer_orbits(safedir, datadir, debug=debug)

        return pd.DataFrame.from_records(records)

    @staticmethod
    def _check_existing(safedir, datadir, bursts, catalog):
        """Raise for an empty file (utils_files.exists) among the files of the bursts already in ``datadir`` and
        the orbit files in ``safedir`` and ``datadir``."""
        from .utils_S1 import measurement_path
        for burst in bursts:
            full_burst_id = ((catalog.get(burst) or {}).get('burst') or {}).get('fullBurstID')
            if not full_burst_id:
                continue
            target_dir = os.path.join(datadir, full_burst_id)
            measurement_path(os.path.join(target_dir, 'measurement'), burst)
            for sub in ('annotation', 'calibration', 'noise'):
                exists(os.path.join(target_dir, sub, f'{burst}.xml'))
        for eof in glob(os.path.join(safedir, '*.EOF')):
            exists(eof)
            exists(os.path.join(datadir, os.path.basename(eof)))

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
    def _process_burst(cls, burst, properties, datadir, safe_index, skip_exist, debug):
        import xmltodict

        # the ASF catalog record of the burst, None for a name the catalog does not return
        if properties is None:
            raise ValueError(f'burst {burst} is not in the ASF catalog')
        full_burst_id = (properties.get('burst') or {}).get('fullBurstID')
        if not full_burst_id:
            raise ValueError(f'the ASF catalog record of {burst} is not a Sentinel-1 burst')

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
        with open(src_ann, 'rb') as f:
            annotation = xmltodict.parse(f.read())['product']
        burst_index, lines_per_burst, samples_per_burst = cls._find_burst_index(
            annotation, datetime_str, burst,
        )

        # the burst directory is the catalog's fullBurstID, as ASF.download names it; its first field is the path
        from .utils_S1 import measurement_path
        path = int(full_burst_id.split('_')[0])

        # Plan target paths. The burst is stored as <burst>.nc; bursts translated before keep their <burst>.tiff.
        target_dir = os.path.join(datadir, full_burst_id)
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

        # Count bursts in the source annotation. 1 burst → convert the whole TIFF.
        # >1 → real burst extraction.
        burst_list = annotation['swathTiming']['burstList']['burst']
        if not isinstance(burst_list, list):
            burst_list = [burst_list]
        n_bursts_in_safe = len(burst_list)

        # the XMLs of the burst, as the downloads write them (utils_S1.burst_xmls), built before anything is
        # written; a single-burst SAFE gives the same files again, with the byteOffset of the .nc written below
        from .utils_S1 import burst_xmls, PAIRS_TIFF_OFFSET, write_slc
        from .utils_files import write_file
        with open(src_noise, 'rb') as f:
            noise = xmltodict.parse(f.read())['noise']
        with open(src_cal, 'rb') as f:
            calibration = xmltodict.parse(f.read())['calibration']
        xmls = burst_xmls(annotation, noise, calibration, burst_index, PAIRS_TIFF_OFFSET)

        if n_bursts_in_safe == 1:
            with open(src_meas, 'rb') as f:
                write_slc(f.read(), tgt_meas)
            mode = 'converted'
        else:
            cls._extract_multiburst(src_meas, tgt_meas, burst_index, lines_per_burst, samples_per_burst)
            mode = f'extracted (burst {burst_index + 1}/{n_bursts_in_safe})'
        # each written through a temporary file renamed when complete
        for tgt, content in zip((tgt_ann, tgt_noise, tgt_cal), xmls):
            write_file(tgt, content)

        if debug:
            print(f'  {mode} {burst} -> {target_dir}')
        return {'burst': burst, 'status': 'created', 'path': path,
                'burstId': int(burst_id_num), 'subswath': subswath,
                'target_dir': target_dir}

    # ---- multi-burst SAFE: extract the burst's rows of the TIFF ----

    @classmethod
    def _extract_multiburst(cls, src_meas, tgt_meas, burst_index, lines_per_burst, samples_per_burst):
        # the requested burst's rows of the multi-burst TIFF, as a standalone single-burst TIFF in memory
        import io
        from tifffile import TiffFile
        from .utils_S1 import write_slc
        tiff_bytes = cls._burst_tiff_bytes(src_meas, burst_index, lines_per_burst)
        with TiffFile(io.BytesIO(tiff_bytes)) as tif:
            actual_lines, actual_samples = tif.pages[0].shape
        if actual_lines != lines_per_burst or actual_samples != samples_per_burst:
            raise RuntimeError(
                f'TIFF dimension mismatch after extraction: got '
                f'{actual_lines}x{actual_samples}, expected '
                f'{lines_per_burst}x{samples_per_burst}'
            )
        write_slc(tiff_bytes, tgt_meas)

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
            exists(found[0])
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
            exists(ann_glob[0])
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
        ID encodes only seconds while ``azimuthTime`` has microseconds. No burst
        at that second raises an error naming the closest ``azimuthTime``.
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

        diffs = [abs((datetime.strptime(b['azimuthTime'], '%Y-%m-%dT%H:%M:%S.%f') - target).total_seconds())
                 for b in burst_list]
        idx = min(range(len(diffs)), key=diffs.__getitem__)
        raise ValueError(
            f'no burst at {datetime_str} in the SAFE annotation for {burst}: '
            f'the closest azimuthTime is {burst_list[idx]["azimuthTime"]}'
        )

    @staticmethod
    def _is_present(path):
        # an empty file raises
        return exists(path) and os.path.isfile(path)

    @staticmethod
    def _transfer(src, tgt):
        # copied with its modification time (shutil.copy2) to a temporary file renamed when complete
        # (utils_files.write_atomic), so an interrupted copy leaves no target file that would pass as data; the
        # rename replaces a previous file or link
        import shutil
        from .utils_files import write_atomic
        write_atomic(tgt, lambda tmp: shutil.copy2(src, tmp))

    @classmethod
    def _transfer_orbits(cls, safedir, datadir, debug):
        for eof in glob(os.path.join(safedir, '*.EOF')):
            tgt = os.path.join(datadir, os.path.basename(eof))
            if cls._is_present(tgt):
                continue
            exists(eof)
            cls._transfer(eof, tgt)
            if debug:
                print(f'  orbit copied: {os.path.basename(eof)}')
