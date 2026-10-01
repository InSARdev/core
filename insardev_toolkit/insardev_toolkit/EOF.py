# ----------------------------------------------------------------------------
# insardev_toolkit
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_toolkit directory for license terms.
# ----------------------------------------------------------------------------
from .progressbar_joblib import progressbar_joblib

class EOF(progressbar_joblib):

    import pandas as pd
    from datetime import timedelta

    http_timeout = 30
    #https://s1orbits.insar.dev/S1A/2014/04/07/index.csv
    #https://s1orbits.insar.dev/S1A/2014/04/07/S1A_OPER_AUX_POEORB_OPOD_20210301T130653_V20140406T225944_20140408T005944.EOF.zip
    orbits_url = 'https://s1orbits.insar.dev/{mission}/{year}/{month:02}/{day:02}/'
    # see _select_orbit.py in sentineleof package
    #Orbital period of Sentinel-1 in seconds
    #T_ORBIT = (12 * 86400.0) / 175.0
    # ESA orbits doe not follow the specification and some orbits are missed
    # example scene: S1A_IW_SLC__1SDV_20240505T002709_20240505T002733_053728_06870F_8D20
    #orbit_offset_start = timedelta(seconds=(12 * 86400.0) // 175.0 + 60)
    # less strict rule allows to find the required orbits.
    # The Sentinel-1 scanners of the preprocessors take the same margins, read at the scan (utils_eof.margins)
    orbit_offset_start = timedelta(seconds=3600)
    orbit_offset_end = timedelta(seconds=300)

    @staticmethod
    def download(basedir: str, scenes: list | pd.DataFrame,
                        n_jobs: int = 8, joblib_backend='loky', skip_exist: bool = True,
                        retries: int = 30, timeout_second: float = 3,
                        min_rate='100KB', min_rate_window: float = 60):
        """
        Downloads orbit files corresponding to the specified Sentinel-1 scenes.

        Each scene time takes the orbit file of its mission that covers it (orbit_offset_start after the validity
        start, orbit_offset_end before the stop), a precise orbit (POEORB) first, then the newest production
        (utils_eof.select); the Sentinel-1 scan of the data directory selects the same way. An orbit file is
        checked in memory and written under a temporary name, renamed only when it is complete, so a failed
        download leaves no orbit file behind.

        Parameters
        ----------
        basedir : str
            The directory where the downloaded orbit files will be saved.
        scenes : list or pandas.DataFrame
            List of scene identifiers or a DataFrame containing scenes for which the orbits are to be downloaded.
        n_jobs : int, optional
            The number of concurrent download jobs. Default is 8.
        joblib_backend : str, optional
            The backend for parallel processing. Default is 'loky'.
        skip_exist : bool, optional
            If True, downloads the orbits of the scenes without one, and the precise orbit (POEORB) of the scenes
            that have a restituted orbit (RESORB) once the POEORB is published, about 20 days after the acquisition;
            a RESORB is kept, with a note, while no POEORB is published. When the orbit server is unreachable (no
            connection, DNS failure, a proxy that is not reached, timeout) and every scene has an orbit, the check
            for POEORB is skipped with a warning; an HTTP error, also from a proxy, raises. The files already in
            basedir are kept, and an empty one raises. If False, deletes every *.EOF file in basedir and downloads
            the orbits of all the scenes. Default is True.
        retries : int, optional
            The number of attempts of each orbit index and orbit file request, the first one included; 0 makes one
            attempt and no retry, as 1 does (HTTP.attempts). Retried are an orbit server that is not reached
            (connection error, DNS failure, a proxy that is not reached, timeout), the HTTP statuses 408, 429 and
            5xx (also from a proxy), and every other failure (another HTTP status, a response that is not a valid
            orbit index or orbit file, which may be a truncated transfer). A failure that a retry cannot change
            fails at its first attempt, logged as not retried (HTTP.final): the HTTP statuses 400, 401, 403, 404,
            405, 407, 409, 410 and 422, also from a proxy, a URL that cannot be requested, a redirect loop, and an
            orbit index that lists no orbit file covering a scene. Default is 30.
        timeout_second : float, optional
            Seconds between the attempts. Default is 3.
        min_rate : str or float, optional
            Bytes per second an orbit index or orbit file download must keep, a size string such as '100KB' or a
            number, averaged over min_rate_window seconds from its first byte, or it is retried on a new connection
            (HTTP.read_body); the rate every one of the n_jobs parallel downloads must reach. Default '100KB'.
        min_rate_window : float, optional
            Seconds over which the rate is averaged. Default is 60.

        Returns
        -------
        pandas.Series
            A Series containing the names of the downloaded orbit files, also after the warning for an orbit
            server that is unreachable while the orbit files download.
    
        Raises
        ------
        Every date is requested before the errors are judged. An error keeps its exception type and names the
        mission, the date and the URL.

        ValueError
            If an invalid scenes argument or a negative retries is provided or no suitable orbit files are found
            (utils_eof.OrbitNotFound, a ValueError, when an orbit index lists no orbit file covering a scene).
        requests.HTTPError
            If the orbit server has no orbit index or orbit file for a mission and date, or refuses it (an HTTP
            error status), for every date, including the dates that only seek the POEORB of a RESORB.
        HTTP.ProxyStatusError (a requests.exceptions.ProxyError)
            If a proxy answers with an HTTP error status (such as 407), for every date, as an HTTP error.
        requests.ConnectionError, requests.Timeout
            If the orbit server is unreachable and a scene is still without an orbit.
        """
        import pandas as pd
        import requests
        import os
        import re
        import glob
        from datetime import datetime
        import time
        import joblib
        import zipfile
        from io import BytesIO
        import xmltodict
        from tqdm.auto import tqdm
        from . import utils_eof
        from .HTTP import final, attempts, answer, send, read_body, ProxyStatusError
        from .utils_S1 import _write_buffer
        from .utils_files import exists

        # a negative retries raises before any file is deleted
        attempts(retries)

        # create the directory if needed
        os.makedirs(basedir, exist_ok=True)

        if skip_exist:
            # the orbit files kept in basedir; an empty one raises before any download
            for orbit in glob.glob('*.EOF', root_dir=basedir):
                exists(os.path.join(basedir, orbit))
        else:
            orbits = glob.glob('*.EOF', root_dir=basedir)
            #print ('orbits', orbits)
            for orbit in orbits:
                os.remove(os.path.join(basedir, orbit))
    
        # an S1 or Nisar object holds its records in .df, accept it as the records themselves.
        # Checked on .df rather than on a to_dataframe() method, which Xarray objects have too.
        if not isinstance(scenes, pd.DataFrame) and isinstance(getattr(scenes, 'df', None), pd.DataFrame):
            scenes = scenes.df

        if isinstance(scenes, pd.DataFrame):
            if skip_exist:
                # scenes without orbits, and scenes with a restituted orbit, which a precise orbit replaces
                # when it is published; the precise flag marks the scenes that take a precise orbit only
                precise = scenes.orbit.map(lambda name: isinstance(name, str)
                                           and utils_eof.parse(name).product == 'RESORB').values
                selected = scenes.orbit.isnull().values | precise
                df_data = scenes[['mission', 'startTime']].assign(precise=precise)[selected].values
            else:
                # process all the scenes
                df_data = scenes[['mission', 'startTime']].assign(precise=False).values
        elif isinstance(scenes, list):
            df_data = [(scene.split('_')[0], datetime.strptime(scene.split('_')[5],'%Y%m%dT%H%M%S'), False)
                       for scene in scenes]
        else:
            raise ValueError(f'Expected secenes argument is list or Pandas DataFrame')
        # nothing to do
        if len(df_data) == 0:
            return
        df = pd.DataFrame(df_data, columns=['mission', 'startTime', 'precise'])
        df['date'] = df['startTime'].dt.date
        # every distinct scene time of a date takes the file that covers it, as the scan of the data directory
        # selects per scene (utils_eof.select), so that the scan finds a covering orbit for each of them;
        # a date takes any orbit when one of its scenes has none
        df = df.groupby(['date', 'mission']).agg(times=('startTime', lambda times: sorted(set(times))),
                                                 precise=('precise', 'all')).reset_index()

        def unreachable(error):
            # the orbit server is not reached: no connection, DNS failure, a proxy that is not reached, timeout. A
            # proxy that answers with an HTTP status is reached, as a server with an HTTP error is
            return isinstance(error, (requests.ConnectionError, requests.Timeout)) and answer(error) is None

        def named(e, text, **kwargs):
            # the same exception type, with a plain message that names the request and its URL, which a joblib
            # worker returns in full
            return type(e)(f'{text}: {e}', **kwargs)

        def get(url, request, read):
            # read(body) of the request. An HTTP error status, of the orbit server or of a proxy, raises with
            # the message it returned, or its status line (HTTP.send); a server that is not reached raises its
            # connection error, and a body slower than min_rate is cut (HTTP.read_body) and retried.
            # Every error keeps its exception type and names the request and the URL
            try:
                response = send(requests.get, url, what=f'{request} cannot be downloaded from {url}',
                                stream=True, timeout=EOF.http_timeout)
            except (requests.HTTPError, ProxyStatusError):
                raise
            except Exception as e:
                raise named(e, f'{request} cannot be downloaded from {url}') from None
            with response:
                try:
                    body = read_body(response, min_rate, min_rate_window)
                except Exception as e:
                    raise named(e, f'{request} cannot be downloaded from {url}') from None
                try:
                    return read(body)
                except Exception as e:
                    raise named(e, f'{request} from {url} cannot be used') from None

        def download_index(mission, date, times, orbit_offset_start, orbit_offset_end, precise):
            url = EOF.orbits_url.format(mission=mission,
                                       year=date.year,
                                       month=date.month,
                                       day=date.day)
            #print ('url', url + 'index.csv')
            index = url + 'index.csv'
            # a line that is not an orbit file name raises, as a data parse error
            orbits = get(index, f'The orbit index for {mission} on {date}',
                         lambda body: [utils_eof.parse(line) for line in body.decode().splitlines()
                                       if line.strip()])
            if precise:
                # a restituted orbit is replaced by a precise one only, and is kept while none is published
                orbits = [orbit for orbit in orbits if orbit.product == 'POEORB']
            # the file covering each time: a precise orbit first, then the most recent production;
            # None keeps the restituted orbit of the time
            names = []
            for time in times:
                name = utils_eof.select(orbits, mission, time, orbit_offset_start, orbit_offset_end)
                if name is None and not precise:
                    # downloading is not possible: the index was read and does not list it, which a retry cannot change
                    raise utils_eof.OrbitNotFound(f'Orbit product not found for mission {mission} and timestamp '
                                                  f'{time} in {index}')
                names.append(name)
            return [(name, url) for name in dict.fromkeys(names)]
    
        # TODO: unzip files
        def download_orbit(basedir, url, request):
            filename = os.path.join(basedir, os.path.basename(os.path.splitext(url)[0]))
            #print ('url', url, 'filename', filename)
            def write(body):
                # checked in memory, then written through a temporary file that is renamed when complete, so that
                # a response that is not a valid orbit file, or a failed write, leaves no file for the scan to take
                with zipfile.ZipFile(BytesIO(body), 'r') as zip_in:
                    zip_files = zip_in.namelist()
                    if len(zip_files) == 0:
                        raise Exception('ERROR: Downloaded file is empty zip archive.')
                    if len(zip_files) > 1:
                        raise Exception('NOTE: Downloaded zip archive includes multiple files.')
                    # extract specific file content
                    orbit_content = zip_in.read(zip_files[0])
                # check XML validity
                xmltodict.parse(orbit_content)
                _write_buffer(BytesIO(orbit_content), filename)
            get(url, request, write)

        def download_with_retry(func, retries, timeout_second, *args, **kwargs):
            # the result and None, or None and the error of the last attempt, so that every request of a stage
            # has its outcome before the errors are judged; retries=0 makes one attempt, as retries=1 (HTTP.attempts).
            # A failure that a retry cannot change (HTTP.final) ends the attempts of the request at once
            n = attempts(retries)
            for retry in range(n):
                try:
                    return func(*args, **kwargs), None
                except Exception as e:
                    stop = final(e)
                    print(f'ERROR: download attempt {retry+1} failed{" (not retried)" if stop else ""}: {e}')
                    if stop or retry + 1 >= n:
                        return None, e
                time.sleep(timeout_second)

        def skip_unreachable(outcomes, rows, lacking, skipped):
            # an error other than an unreachable server raises, the first in date order, and names its date.
            # Otherwise an unreachable server raises when a date is still without an orbit (lacking), and skips
            # the stage for the dates (rows) with a warning when every scene has an orbit. True when skipped
            errors = [error for _, error in outcomes if error is not None]
            for error in errors:
                if not unreachable(error):
                    raise error
            if not errors:
                return False
            e = errors[0]
            failure = f'{type(e).__name__}: {e}'
            def names(rows):
                return ', '.join(dict.fromkeys(f'{row.mission} on {row.date}' for row in rows.itertuples()))
            if len(lacking):
                raise type(e)(f'The orbit server is unreachable, the orbits for {names(lacking)} '
                              f'cannot be downloaded ({failure})') from e
            print(f'WARNING: The orbit server is unreachable ({failure}), {skipped} for {names(rows)} is skipped, '
                  f'their restituted orbits (RESORB) are kept.')
            return True

        # download orbits index files and detect the orbits.
        # joblib reads the class file from disk,
        # and to allow modification of orbit_offset_* on the fly, add them to the arguments
        with EOF.progressbar_joblib(tqdm(desc='Downloading Orbits List'.ljust(25), total=len(df))) as progress_bar:
            outcomes = joblib.Parallel(n_jobs=n_jobs, backend=joblib_backend)(joblib.delayed(download_with_retry)\
                                    (download_index, retries, timeout_second,
                                    mission=scene.mission,
                                    date=scene.date,
                                    times=scene.times,
                                    orbit_offset_start=EOF.orbit_offset_start,
                                    orbit_offset_end=EOF.orbit_offset_end,
                                    precise=scene.precise) for scene in df.itertuples())
        # nothing is downloaded yet: the dates with a scene without an orbit lack it
        if skip_unreachable(outcomes, df, df[~df.precise], 'the check for the precise orbits (POEORB)'):
            return None
        selected = [files for files, _ in outcomes]
        for scene, files in zip(df.itertuples(), selected):
            if any(name is None for name, _ in files):
                print(f'NOTE: No precise orbit (POEORB) covering the scenes is published yet for {scene.mission} '
                      f'on {scene.date}, its restituted orbit (RESORB) is kept.')
        # convert to dataframe for processing, with the dates that select each file
        dates = pd.DataFrame([(name, url, scene.mission, scene.date, scene.precise)
                              for scene, files in zip(df.itertuples(), selected)
                              for name, url in files if name is not None],
                             columns=['orbit', 'url', 'mission', 'date', 'precise'])
        # a file selected for several dates is downloaded once
        orbits = dates.drop_duplicates('orbit').reset_index(drop=True)
        # nothing to do
        if len(orbits) == 0:
            return
        
        # download orbits index files and detect the orbits
        with EOF.progressbar_joblib(tqdm(desc='Downloading Orbit Files'.ljust(25), total=len(orbits))) as progress_bar:
            outcomes = joblib.Parallel(n_jobs=n_jobs, backend=joblib_backend)(joblib.delayed(download_with_retry)\
                                    (download_orbit, retries, timeout_second,
                                    basedir=basedir,
                                    url=orbit.url + orbit.orbit,
                                    request=f'The orbit file {orbit.orbit} for {orbit.mission} on {orbit.date}')
                                    for orbit in orbits.itertuples())
        # the dates whose files are not on disk now; those with a scene without an orbit lack it
        missing = dates[[not exists(os.path.join(basedir, os.path.splitext(name)[0])) for name in dates.orbit]]
        if skip_unreachable(outcomes, missing, missing[~missing.precise],
                            'the download of the precise orbits (POEORB)'):
            # the files downloaded
            return orbits['orbit'][[error is None for _, error in outcomes]].reset_index(drop=True)
        return orbits['orbit']
