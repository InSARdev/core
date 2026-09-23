# ----------------------------------------------------------------------------
# insardev_toolkit
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_toolkit directory for license terms.
# ----------------------------------------------------------------------------
import io
import os
import requests
import zipfile
import shutil
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm.auto import tqdm

class HTTPRangeReader(io.RawIOBase):
    """
    A RawIOBase wrapper that does HTTP Range requests on demand,
    reusing a single Session + HTTPAdapter for keep-alive and retries,
    and feeding a tqdm bar as it reads.
    """
    def __init__(self, url: str, max_retries=3, backoff=0.5):
        self.url = url
        # fetch total size once
        head = requests.head(url, allow_redirects=True)
        head.raise_for_status()
        self.total_size = int(head.headers.get("Content-Length", 0))
        self.pos = 0

        # set up session + retries
        self.session = requests.Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=backoff,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # progress bar for actual bytes read
        self.bar = tqdm(
            total=self.total_size,
            unit="B", unit_scale=True, unit_divisor=1024,
            desc=os.path.basename(url),
        )

    def readinto(self, b: bytearray) -> int:
        if self.pos >= self.total_size:
            return 0
        length = len(b)
        end = min(self.pos + length - 1, self.total_size - 1)
        headers = {"Range": f"bytes={self.pos}-{end}"}
        r = self.session.get(self.url, headers=headers, stream=True)
        r.raise_for_status()
        # read directly into buffer
        n = r.raw.readinto(b)
        self.pos += n
        self.bar.update(n)
        return n

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.total_size + offset
        else:
            raise ValueError("Invalid whence")
        self.pos = max(0, min(self.pos, self.total_size))
        return self.pos

    def tell(self):
        return self.pos

    def readable(self):
        return True

    def seekable(self):
        return True

    def close(self):
        self.bar.close()
        return super().close()

def unzip(url: str, target_dir: str|None = None, buffer_size: int = 4 << 20):
    """
    Stream-download unpacking remote ZIP content via HTTP Range requests.
    Skips already-extracted files (filling the bar by the compressed size).
    Works well with large files on Zenodo not hummering the server.
    """
    if target_dir is None:
        target_dir = os.path.basename(url).split('.')[0]
    else:
        target_dir = os.path.expanduser(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    raw = HTTPRangeReader(url)
    buf = io.BufferedReader(raw, buffer_size=buffer_size)

    with zipfile.ZipFile(buf) as zf:
        for info in zf.infolist():
            # skip macOS metadata folder contents
            # it is empty and useless but requires extra long time to extract
            if info.filename.startswith("__MACOSX/"):
                continue
            
            out_path = os.path.join(target_dir, info.filename)

            # create empty directories listed in zip
            if info.is_dir() or info.filename.endswith("/"):
                os.makedirs(out_path, exist_ok=True)
                continue

            # skip if already extracted and complete
            if os.path.exists(out_path) and os.path.getsize(out_path) == info.file_size:
                # account compressed bytes for progress
                raw.bar.update(info.compress_size)
                continue

            # extract creating non-empty unlisted directories for all files
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with zf.open(info) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst, length=64 << 10)

    # downloaded size does not include zip header, central directory, etc.
    # to make progress bar complete, we need to add the remaining size
    remaining = raw.total_size - raw.bar.n
    if remaining > 0:
        raw.bar.update(remaining)
    raw.close()

class NotFound(Exception):
    """The server states that the resource does not exist: the only outcome of fetch() that is not retried."""


# body signatures of the payloads the downloaders fetch
MAGIC_ZIP = (b'PK\x03\x04',)
MAGIC_GZIP = (b'\x1f\x8b',)
MAGIC_TIFF = (b'II*\x00', b'MM\x00*')
MAGIC_HDF5 = (b'\x89HDF',)
MAGIC_IMAGE = (b'\x89PNG', b'\xff\xd8\xff', b'GIF8', b'RIFF')

_sessions = {}


def _session():
    """One keep-alive session per process and thread, reused by every fetch() call in it."""
    import threading
    key = (os.getpid(), threading.get_ident())
    if key not in _sessions:
        _sessions[key] = requests.Session()
    return _sessions[key]


def fetch(url: str, session=None, headers=None, magic=None, retries: int = 30, timeout_second: float = 3,
          timeout=(10, 300), min_rate='100KB', min_rate_window: float = 60,
          debug: bool = False) -> bytes:
    """
    Download a whole response body into memory.

    Every failure is retried: connection errors, HTTP errors, a body shorter than its Content-Length, a body
    that is not the expected payload, and a transfer that stays slower than the minimum rate. A read timeout
    alone never fires while bytes keep arriving, however slowly, so a trickling transfer is cut by the rate check
    instead, made after every network read, and the next attempt starts a new connection. The rate is measured
    from the first byte of the body: a server that stays silent before it, as a proxy does while it fetches the
    data upstream, is bounded by the read timeout. A transfer that keeps the minimum rate is never cut, however
    long it takes.

    Only an explicit "does not exist" is final and raises NotFound without retries: HTTP 404 or 410, and an
    HTML page served with HTTP 200 in place of the expected payload (a soft 404 page).

    Parameters
    ----------
    url : str
        URL to download.
    session : requests.Session, optional
        Session to use. A per-process keep-alive session by default.
    headers : dict, optional
        Request headers.
    magic : tuple of bytes, optional
        Accepted leading bytes of the body, e.g. MAGIC_ZIP. No check when None.
    retries : int, optional
        Number of attempts. Default 30.
    timeout_second : float, optional
        Seconds between attempts. Default 3.
    timeout : tuple, optional
        (connect, read) timeouts in seconds. Default (10, 300): a proxy can stay silent for minutes while it
        fetches the data upstream.
    min_rate : str or float, optional
        Minimum transfer rate in bytes per second, a size string such as '100KB' or a number, checked over each
        min_rate_window. Default '100KB'.
    min_rate_window : float, optional
        Seconds over which the rate is measured. Default 60.
    debug : bool, optional
        Print every failed attempt. Default False.

    Returns
    -------
    bytes
        The response body.

    Raises
    ------
    NotFound
        The resource does not exist.
    Exception
        The last error once all attempts have failed.
    """
    import time

    if retries < 1:
        raise ValueError(f'ERROR: retries must be at least 1, got {retries}')
    if isinstance(min_rate, str):
        from dask.utils import parse_bytes
        min_rate = parse_bytes(min_rate)
    session = session or _session()
    for attempt in range(retries):
        try:
            with session.get(url, headers=headers, stream=True, timeout=timeout) as response:
                if response.status_code in (404, 410):
                    raise NotFound(f'HTTP {response.status_code}: {url}')
                response.raise_for_status()
                body = bytearray()
                window_t = window_n = None
                # read1() returns what one network read brings, up to 64 KB, so the rate check never waits for
                # more; urllib3 before 2.0 has no read1() and its read() waits for the whole 64 KB
                read1 = getattr(response.raw, 'read1', None)
                while True:
                    chunk = read1(64 * 1024, decode_content=True) if read1 else response.raw.read(64 * 1024, decode_content=True)
                    if not chunk:
                        break
                    body += chunk
                    now = time.monotonic()
                    if window_t is None:
                        # the rate is measured from the first byte, the silence before it is the read timeout's
                        window_t, window_n = now, 0
                        continue
                    window_n += len(chunk)
                    if now - window_t >= min_rate_window:
                        rate = window_n / (now - window_t)
                        if rate < min_rate:
                            raise TimeoutError(f'transfer rate {rate / 1000:.1f} KB/s is below {min_rate / 1000:.0f} KB/s')
                        window_t, window_n = now, 0
                # a transparently decoded body has another length than the one announced
                encoding = response.headers.get('Content-Encoding', 'identity').lower()
                expected = response.headers.get('Content-Length')
                if expected is not None and encoding == 'identity' and len(body) != int(expected):
                    raise IOError(f'truncated body: {len(body)} of {expected} bytes')
                if magic is not None and not bytes(body[:8]).startswith(tuple(magic)):
                    if 'text/html' in response.headers.get('Content-Type', '').lower():
                        raise NotFound(f'HTML page instead of the expected payload: {url}')
                    raise IOError(f'unexpected payload {bytes(body[:8])!r}')
                return bytes(body)
        except NotFound:
            raise
        except Exception as e:
            if debug or attempt + 1 == retries:
                print(f'ERROR: download attempt {attempt + 1}/{retries} failed for {url}: {e}')
            if attempt + 1 == retries:
                raise
            time.sleep(timeout_second)


def download(url: str, dst: str, chunk_size: int = 2**20):
    import os
    import requests
    from tqdm.auto import tqdm

    downloaded = os.path.getsize(dst) if os.path.exists(dst) else 0
    head = requests.head(url, allow_redirects=True)
    head.raise_for_status()
    expected = int(head.headers.get("content-length", 0))
    #print ('downloaded', downloaded, 'expected', expected)
    if downloaded and expected and downloaded >= expected:
        print(f"{dst} is already fully downloaded.")
        return
    if os.path.exists(dst):
        print(f'{dst} is incompletely downloaded, removing it.')
        os.remove(dst)
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get('content-length', 0))
    with open(dst, 'wb') as f, \
         tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
              desc=os.path.basename(dst)) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            f.write(chunk)
            bar.update(len(chunk))
