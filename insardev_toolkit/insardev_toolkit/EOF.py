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
            List of scene identifiers, a DataFrame containing scenes, or the table of ASF.search (platform, startTime)
            for which the orbits are to be downloaded.
        n_jobs : int, optional
            The number of concurrent download jobs, at most two per CPU this process may use (a larger number is
            capped, with a note). Negative values follow joblib: -1 all CPUs, -2 all but one, and so on; 0 raises
            ValueError. Default is 8.
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
        import os
        import glob
        from datetime import datetime
        import joblib
        from io import BytesIO
        import xmltodict
        from tqdm.auto import tqdm
        from . import utils_eof
        from .HTTP import attempts
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

        # the ASF search table (ASF.search): the mission from its platform (SENTINEL-1A is S1A), the start time in UTC; its orbit column is
        # the absolute orbit number, not an orbit file, so with skip_exist each scene takes the held orbit file that covers it
        if isinstance(scenes, pd.DataFrame) and 'mission' not in scenes.columns and {'platform', 'startTime'} <= set(scenes.columns):
            platform = scenes['platform'].astype(str).str.upper()
            if not platform.str.match(r'^SENTINEL-1[A-D]$').all():
                raise ValueError(f'ERROR: not a Sentinel-1 platform: {sorted(set(platform[~platform.str.match(r"^SENTINEL-1[A-D]$")]))}')
            times = pd.to_datetime(scenes['startTime'], utc=True).dt.tz_localize(None)
            scenes = pd.DataFrame({'mission': ('S1' + platform.str[-1]).values, 'startTime': times.values})
            if skip_exist:
                found = utils_eof.parse_files(glob.glob('*.EOF', root_dir=basedir))
                scenes['orbit'] = [utils_eof.select(found, mission, when) for mission, when in zip(scenes['mission'], scenes['startTime'])]

        # a list of scene names is the records of its scenes; with skip_exist each takes the orbit file of basedir that
        # covers it (the scan of the data directory, utils_eof.select), as the orbit column of a DataFrame does
        if isinstance(scenes, list):
            scenes = pd.DataFrame([(scene.split('_')[0], datetime.strptime(scene.split('_')[5], '%Y%m%dT%H%M%S'))
                                   for scene in scenes], columns=['mission', 'startTime'])
            if skip_exist:
                found = utils_eof.parse_files(glob.glob('*.EOF', root_dir=basedir))
                scenes['orbit'] = [utils_eof.select(found, mission, when)
                                   for mission, when in zip(scenes['mission'], scenes['startTime'])]

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

        def download_index(mission, date, times, orbit_offset_start, orbit_offset_end, precise):
            url, orbits = EOF._index(mission, date, min_rate, min_rate_window)
            index = url + 'index.csv'
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
                orbit_content = EOF._unzip(body)
                # check XML validity
                xmltodict.parse(orbit_content)
                _write_buffer(BytesIO(orbit_content), filename)
            EOF._get(url, request, write, min_rate, min_rate_window)

        def skip_unreachable(outcomes, rows, lacking, skipped):
            # an error other than an unreachable server raises, the first in date order, and names its date.
            # Otherwise an unreachable server raises when a date is still without an orbit (lacking), and skips
            # the stage for the dates (rows) with a warning when every scene has an orbit. True when skipped
            errors = [error for _, error in outcomes if error is not None]
            for error in errors:
                if not EOF._unreachable(error):
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

        from .HTTP import download_jobs
        from .utils_pool import release_workers
        n_jobs = download_jobs(n_jobs)
        # download orbits index files and detect the orbits.
        # joblib reads the class file from disk,
        # and to allow modification of orbit_offset_* on the fly, add them to the arguments
        with EOF.progressbar_joblib(tqdm(desc='Downloading Orbits List'.ljust(25), total=len(df))) as progress_bar:
            outcomes = joblib.Parallel(n_jobs=n_jobs, backend=joblib_backend)(joblib.delayed(EOF._with_retry)\
                                    (download_index, retries, timeout_second,
                                    mission=scene.mission,
                                    date=scene.date,
                                    times=scene.times,
                                    orbit_offset_start=EOF.orbit_offset_start,
                                    orbit_offset_end=EOF.orbit_offset_end,
                                    precise=scene.precise) for scene in df.itertuples())
        # the idle joblib workers keep their memory: stop them, the next parallel step starts its own
        release_workers()
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
            outcomes = joblib.Parallel(n_jobs=n_jobs, backend=joblib_backend)(joblib.delayed(EOF._with_retry)\
                                    (download_orbit, retries, timeout_second,
                                    basedir=basedir,
                                    url=orbit.url + orbit.orbit,
                                    request=f'The orbit file {orbit.orbit} for {orbit.mission} on {orbit.date}')
                                    for orbit in orbits.itertuples())
        # the idle joblib workers keep their memory: stop them, the next parallel step starts its own
        release_workers()
        # the dates whose files are not on disk now; those with a scene without an orbit lack it
        missing = dates[[not exists(os.path.join(basedir, os.path.splitext(name)[0])) for name in dates.orbit]]
        if skip_unreachable(outcomes, missing, missing[~missing.precise],
                            'the download of the precise orbits (POEORB)'):
            # the files downloaded
            return orbits['orbit'][[error is None for _, error in outcomes]].reset_index(drop=True)
        return orbits['orbit']

    # the requests of the orbit server, shared by EOF.download and the Sentinel-1 burst downloaders

    @staticmethod
    def _unreachable(error):
        """
        Private helper of the Sentinel-1 downloaders: whether the orbit server was not reached.

        Parameters
        ----------
        error : Exception
            The error of a request.

        Returns
        -------
        bool
            True for no connection, a DNS failure, a proxy that is not reached or a timeout. A proxy that answers
            with an HTTP status is reached, as a server with an HTTP error is.
        """
        import requests
        from .HTTP import answer
        return isinstance(error, (requests.ConnectionError, requests.Timeout)) and answer(error) is None

    @staticmethod
    def _named(e, text, **kwargs):
        """
        Private helper of the Sentinel-1 downloaders: the same exception type, with a plain message that names the
        request and its URL, which a joblib worker returns in full.

        Parameters
        ----------
        e : Exception
            The error.
        text : str
            The text put before the error message.
        **kwargs
            Passed to the exception type.

        Returns
        -------
        Exception
            The new exception, not raised.
        """
        return type(e)(f'{text}: {e}', **kwargs)

    @staticmethod
    def _get(url, request, read, min_rate, min_rate_window):
        """
        Private helper of the Sentinel-1 downloaders: one request of the orbit server, read by a function.

        An HTTP error status, of the orbit server or of a proxy, raises with the message it returned, or its status
        line (HTTP.send); a server that is not reached raises its connection error, and a body slower than min_rate
        is cut (HTTP.read_body). Every error keeps its exception type and names the request and the URL.

        Parameters
        ----------
        url : str
            The URL.
        request : str
            The request named in the errors.
        read : callable
            Takes the response body (bytes) and returns the result.
        min_rate : str or float
            Bytes per second the body must keep (HTTP.read_body).
        min_rate_window : float
            Seconds over which the rate is averaged.

        Returns
        -------
        object
            read(body).

        Raises
        ------
        requests.HTTPError, HTTP.ProxyStatusError
            For an HTTP error status.
        Exception
            Another error of the request or of read, of its own type, naming the request and the URL.
        """
        import requests
        from .HTTP import send, read_body, ProxyStatusError
        try:
            response = send(requests.get, url, what=f'{request} cannot be downloaded from {url}',
                            stream=True, timeout=EOF.http_timeout)
        except (requests.HTTPError, ProxyStatusError):
            raise
        except Exception as e:
            raise EOF._named(e, f'{request} cannot be downloaded from {url}') from None
        with response:
            try:
                body = read_body(response, min_rate, min_rate_window)
            except Exception as e:
                raise EOF._named(e, f'{request} cannot be downloaded from {url}') from None
            try:
                return read(body)
            except Exception as e:
                raise EOF._named(e, f'{request} from {url} cannot be used') from None

    @staticmethod
    def _with_retry(func, retries, timeout_second, *args, **kwargs):
        """
        Private helper of the Sentinel-1 downloaders: a request with its attempts.

        Every request of a stage has its outcome before the errors are judged; retries=0 makes one attempt, as
        retries=1 (HTTP.attempts). A failure that a retry cannot change (HTTP.final) ends the attempts at once.

        Parameters
        ----------
        func : callable
            The request, called with args and kwargs.
        retries : int
            The number of attempts.
        timeout_second : float
            Seconds between the attempts.

        Returns
        -------
        tuple
            (result, None), or (None, error) with the error of the last attempt.
        """
        import time
        from .HTTP import final, attempts
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

    @staticmethod
    def _index(mission, date, min_rate, min_rate_window):
        """
        Private helper of the Sentinel-1 downloaders: the orbit index of a mission and day.

        Parameters
        ----------
        mission : str
            The mission (S1A, ...).
        date : datetime.date
            The day.
        min_rate : str or float
            Bytes per second the index download must keep.
        min_rate_window : float
            Seconds over which the rate is averaged.

        Returns
        -------
        tuple
            (url, orbits): the day folder URL (EOF.orbits_url) and the orbit files it lists (utils_eof.Orbit).

        Raises
        ------
        Exception
            The errors of EOF._get; a line that is not an orbit file name raises, as a data parse error.
        """
        from . import utils_eof
        url = EOF.orbits_url.format(mission=mission, year=date.year, month=date.month, day=date.day)
        orbits = EOF._get(url + 'index.csv', f'The orbit index for {mission} on {date}',
                          lambda body: [utils_eof.parse(line) for line in body.decode().splitlines()
                                        if line.strip()],
                          min_rate, min_rate_window)
        return url, orbits

    @staticmethod
    def _unzip(body):
        """
        Private helper of the Sentinel-1 downloaders: the single file of a downloaded orbit zip.

        Parameters
        ----------
        body : bytes
            The zip archive.

        Returns
        -------
        bytes
            The content of its only member.

        Raises
        ------
        Exception
            If the archive is empty or holds several files.
        """
        import zipfile
        from io import BytesIO
        with zipfile.ZipFile(BytesIO(body), 'r') as zip_in:
            zip_files = zip_in.namelist()
            if len(zip_files) == 0:
                raise Exception('ERROR: Downloaded file is empty zip archive.')
            if len(zip_files) > 1:
                raise Exception('NOTE: Downloaded zip archive includes multiple files.')
            return zip_in.read(zip_files[0])

    @staticmethod
    def _orbit_names(scenes, n_jobs=8, joblib_backend='loky', retries=30, timeout_second=3, min_rate='100KB',
                     min_rate_window=60):
        """
        Private helper of the Sentinel-1 burst downloaders: the orbit file URL of every burst.

        The orbit index of every distinct mission and day is read once. Each burst takes the file of its mission
        that covers its time, a precise orbit first, then the newest production (utils_eof.select). A missing
        orbit never raises: it is printed as a WARNING and its burst gets None.

        Parameters
        ----------
        scenes : pandas.DataFrame
            One row per burst, with the columns burst (a name for the messages), mission (S1A, ...) and startTime
            (naive UTC).
        n_jobs, joblib_backend, retries, timeout_second, min_rate, min_rate_window
            As EOF.download.

        Returns
        -------
        list
            Aligned with the rows: the URL of the selected orbit file (.EOF.zip), or None.
        """
        import joblib
        import pandas as pd
        from tqdm.auto import tqdm
        from . import utils_eof
        from .HTTP import download_jobs
        from .utils_pool import release_workers

        rows = [(burst, mission, pd.Timestamp(time))
                for burst, mission, time in zip(scenes['burst'], scenes['mission'], scenes['startTime'])]
        keys = list(dict.fromkeys((mission, time.date()) for _, mission, time in rows))
        n_jobs = download_jobs(n_jobs)
        with EOF.progressbar_joblib(tqdm(desc='Downloading Orbits List'.ljust(25), total=len(keys))) as progress_bar:
            outcomes = joblib.Parallel(n_jobs=n_jobs, backend=joblib_backend)(joblib.delayed(EOF._with_retry)\
                                    (EOF._index, retries, timeout_second, mission, date, min_rate, min_rate_window)
                                    for mission, date in keys)
        # the idle joblib workers keep their memory: stop them, the next parallel step starts its own
        release_workers()
        indexes = dict(zip(keys, outcomes))
        for (mission, date), (_, error) in indexes.items():
            if error is not None:
                names = [burst for burst, m, time in rows if (m, time.date()) == (mission, date)]
                print(f'WARNING: the orbit index for {mission} on {date} cannot be read ({type(error).__name__}: '
                      f'{error}): {len(names)} burst(s) get no orbit: {", ".join(map(str, names))}.')
        urls = []
        for burst, mission, time in rows:
            index, _ = indexes[(mission, time.date())]
            if index is None:
                urls.append(None)
                continue
            url, orbits = index
            name = utils_eof.select(orbits, mission, time)
            if name is None:
                print(f'WARNING: no orbit file in {url}index.csv covers {burst} ({mission} {time}).')
                urls.append(None)
            else:
                urls.append(url + name)
        return urls

    @staticmethod
    def _file_records(url, windows, request, min_rate, min_rate_window):
        """
        Private helper of the Sentinel-1 burst downloaders: the records of an orbit file inside time windows.

        The zip is unpacked and cropped while it is read, so that only the crop leaves the worker.

        Parameters
        ----------
        url : str
            The orbit file URL (.EOF.zip).
        windows : list of tuple
            The (start, stop) windows, naive UTC.
        request : str
            The request named in the errors.
        min_rate, min_rate_window
            As EOF._get.

        Returns
        -------
        list of tuple
            utils_eof.osv_records() of the file in the windows.
        """
        from . import utils_eof
        return EOF._get(url, request, lambda body: utils_eof.osv_records(EOF._unzip(body), windows),
                        min_rate, min_rate_window)

    @staticmethod
    def _orbit_records(scenes, urls=None, n_jobs=8, joblib_backend='loky', retries=30, timeout_second=3,
                       min_rate='100KB', min_rate_window=60):
        """
        Private helper of the Sentinel-1 burst downloaders: the orbit records around every burst.

        Each orbit file is downloaded once, however many bursts use it, and cropped to the windows of its bursts
        (utils_eof.ORBIT_MARGIN + 60 seconds around each start time); utils_eof.set_orbit makes the exact cut. A
        missing orbit never raises: it is printed as a WARNING and its burst gets None.

        Parameters
        ----------
        scenes : pandas.DataFrame
            One row per burst, with the columns burst, mission and startTime (naive UTC), see EOF._orbit_names.
        urls : list, optional
            The orbit file URL of every row, or None; default EOF._orbit_names(scenes).
        n_jobs, joblib_backend, retries, timeout_second, min_rate, min_rate_window
            As EOF.download.

        Returns
        -------
        list
            Aligned with the rows: (source, records), the orbit file name (.EOF) and the crop of its file (the rows
            of one file share one list), or None.
        """
        import os
        import joblib
        import pandas as pd
        from datetime import timedelta
        from tqdm.auto import tqdm
        from . import utils_eof
        from .HTTP import download_jobs
        from .utils_pool import release_workers

        if urls is None:
            urls = EOF._orbit_names(scenes, n_jobs=n_jobs, joblib_backend=joblib_backend, retries=retries,
                                    timeout_second=timeout_second, min_rate=min_rate,
                                    min_rate_window=min_rate_window)
        rows = [(burst, mission, pd.Timestamp(time))
                for burst, mission, time in zip(scenes['burst'], scenes['mission'], scenes['startTime'])]
        # the rows of each orbit file
        groups = {}
        for row, url in enumerate(urls):
            if url is not None:
                groups.setdefault(url, []).append(row)
        margin = timedelta(seconds=utils_eof.ORBIT_MARGIN + 60)
        files = []
        for url, members in groups.items():
            name = os.path.basename(url)
            _, mission, time = rows[members[0]]
            windows = [(rows[row][2] - margin, rows[row][2] + margin) for row in members]
            files.append((url, name, windows, f'The orbit file {name} for {mission} on {time.date()}'))
        n_jobs = download_jobs(n_jobs)
        with EOF.progressbar_joblib(tqdm(desc='Downloading Orbit Files'.ljust(25), total=len(files))) as progress_bar:
            outcomes = joblib.Parallel(n_jobs=n_jobs, backend=joblib_backend)(joblib.delayed(EOF._with_retry)\
                                    (EOF._file_records, retries, timeout_second, url, windows, request, min_rate,
                                     min_rate_window)
                                    for url, name, windows, request in files)
        # the idle joblib workers keep their memory: stop them, the next parallel step starts its own
        release_workers()
        results = [None] * len(rows)
        keep = timedelta(seconds=utils_eof.ORBIT_MARGIN)
        for (url, name, _, _), (records, error) in zip(files, outcomes):
            members = groups[url]
            if error is not None:
                names = [rows[row][0] for row in members]
                print(f'WARNING: the orbit file {name} cannot be downloaded ({type(error).__name__}: {error}): '
                      f'{len(names)} burst(s) get no orbit: {", ".join(map(str, names))}.')
                continue
            source = name.removesuffix('.zip')
            for row in members:
                burst, _, time = rows[row]
                if utils_eof._bracket(records, time - keep, time + keep) is None:
                    print(f'WARNING: the orbit file {name} has no records around {burst} ({time}).')
                    continue
                results[row] = (source, records)
        return results

    @staticmethod
    def _update_bursts(basedir, skip_exist=True, n_jobs=8, joblib_backend='loky', retries=30, timeout_second=3,
                       min_rate='100KB', min_rate_window=60):
        """
        Private helper of ASF.update and CDSE.update: rewrite the orbit stored in the stored Sentinel-1 bursts.

        The bursts are the annotations basedir/*/annotation/S1_*-BURST.xml whose subswath is the one of their folder
        (the IW2 annotation stored in an IW1 or IW3 folder is never changed). Each burst takes the orbit file of its
        mission that covers its start, a precise orbit first, then the newest production (EOF._orbit_names). With
        skip_exist=True only the bursts whose stored source differs from the selected file are rewritten, with
        skip_exist=False every burst with a selected file. The new annotation is utils_eof.set_orbit() of the stored
        bytes, written to a temporary file renamed when complete (utils_files.write_file). A burst without an orbit
        (no published file, a failed download, or records that do not cover it) keeps its file, with a WARNING. The
        .EOF files in basedir are neither read nor deleted; only the orbit server is asked.

        Parameters
        ----------
        basedir : str
            The data directory of the burst folders.
        skip_exist : bool, optional
            True rewrites only the bursts whose stored orbit differs from the selected one, False all. Default True.
        n_jobs, joblib_backend, retries, timeout_second, min_rate, min_rate_window
            As EOF.download.

        Returns
        -------
        pandas.DataFrame or None
            The rewritten bursts (column burst, the annotation file name without .xml) and their new orbit (column
            orbit, the orbit file name), or None when no burst is rewritten.

        Examples
        --------
        >>> EOF._update_bursts('data/')
        """
        import os
        from glob import glob
        import pandas as pd
        import xml.etree.ElementTree as ET
        from . import utils_eof
        from .utils_files import write_file

        paths = []
        for rel in sorted(glob('*/annotation/S1_*-BURST.xml', root_dir=basedir)):
            folder = rel.split(os.sep)[0]
            name = os.path.basename(rel)
            # the IW2 annotation of an IW1 or IW3 burst is auxiliary and never changed
            if name.split('_')[2] == folder.split('_')[-1]:
                paths.append(os.path.join(basedir, rel))
        if not paths:
            print(f'NOTE: no stored Sentinel-1 burst in {basedir}.')
            return None

        contents, bursts, missions, starts, stored = [], [], [], [], []
        for path in paths:
            with open(path, 'rb') as f:
                content = f.read()
            root = ET.fromstring(content)
            contents.append(content)
            bursts.append(os.path.basename(path).removesuffix('.xml'))
            missions.append(root.findtext('adsHeader/missionId'))
            starts.append(pd.Timestamp(root.findtext('swathTiming/burstList/burst/azimuthTime')))
            orbit = utils_eof.stored_orbit(path)
            stored.append(None if orbit is None else orbit[0])
        scenes = pd.DataFrame({'burst': bursts, 'mission': missions, 'startTime': starts})
        urls = EOF._orbit_names(scenes, n_jobs=n_jobs, joblib_backend=joblib_backend, retries=retries,
                                timeout_second=timeout_second, min_rate=min_rate, min_rate_window=min_rate_window)
        selected = [None if url is None else os.path.basename(url).removesuffix('.zip') for url in urls]
        rewrite = [row for row, name in enumerate(selected)
                   if name is not None and (not skip_exist or name != stored[row])]
        # the bursts that hold the selected orbit already (skip_exist=True only)
        held = len([name for name in selected if name is not None]) - len(rewrite)

        records = []
        if rewrite:
            records = EOF._orbit_records(scenes.iloc[rewrite].reset_index(drop=True), [urls[row] for row in rewrite],
                                         n_jobs=n_jobs, joblib_backend=joblib_backend, retries=retries,
                                         timeout_second=timeout_second, min_rate=min_rate,
                                         min_rate_window=min_rate_window)
        rewritten, sources = [], []
        for row, orbit in zip(rewrite, records):
            if orbit is None:
                continue
            source, osvs = orbit
            try:
                new = utils_eof.set_orbit(contents[row], source, osvs)
            except utils_eof.OrbitNotFound as e:
                print(f'WARNING: {bursts[row]} keeps its orbit: {e}')
                continue
            write_file(paths[row], new)
            rewritten.append(bursts[row])
            sources.append(source)
        kept = len(paths) - len(rewritten) - held
        print(f'NOTE: {len(rewritten)} of {len(paths)} stored bursts get a new orbit, {held} hold the selected orbit '
              f'already, {kept} keep theirs (no orbit could be read).')
        if not rewritten:
            return None
        return pd.DataFrame({'burst': rewritten, 'orbit': sources})
