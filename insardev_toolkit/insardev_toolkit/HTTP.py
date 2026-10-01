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

    Every range is read by iter_body() into the caller's buffer, one network read at a time, and cut when the
    transfer stays slower than min_rate; a range that fails so, or in any other way a retry can change (final()),
    is requested again on a new connection, up to retries attempts.
    """
    def __init__(self, url: str, max_retries=3, backoff=0.5, retries: int = 30, timeout_second: float = 3,
                 timeout=(10, 300), min_rate='100KB', min_rate_window: float = 60):
        self.url = url
        self.retries, self.timeout_second, self.timeout = retries, timeout_second, timeout
        self.min_rate, self.min_rate_window = min_rate, min_rate_window
        # a negative retries raises before any request
        attempts(retries)
        # fetch total size once
        head = send(requests.head, url, allow_redirects=True, timeout=timeout)
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
        import time
        length = len(b)
        end = min(self.pos + length - 1, self.total_size - 1)
        headers = {"Range": f"bytes={self.pos}-{end}"}
        view = memoryview(b)
        n = attempts(self.retries)
        for attempt in range(n):
            try:
                count = 0
                with send(self.session.get, self.url, headers=headers, stream=True, timeout=self.timeout) as r:
                    # read into the buffer as the body arrives; a server that ignores the range sends more
                    for chunk in iter_body(r, self.min_rate, self.min_rate_window):
                        if count + len(chunk) > length:
                            raise IOError(f'range response longer than the {length} bytes requested')
                        view[count:count + len(chunk)] = chunk
                        count += len(chunk)
                break
            except Exception as e:
                stop = final(e)
                if stop or attempt + 1 == n:
                    print(f'ERROR: download attempt {attempt + 1}/{n} failed{" (not retried)" if stop else ""} '
                          f'for {self.url}: {returned(e) or e}')
                    raise
                time.sleep(self.timeout_second)
        self.pos += count
        self.bar.update(count)
        return count

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
        # a reader whose size request failed has no bar
        if hasattr(self, 'bar'):
            self.bar.close()
        return super().close()

def unzip(url: str, target_dir: str|None = None, buffer_size: int = 4 << 20, retries: int = 30,
          timeout_second: float = 3, min_rate='100KB', min_rate_window: float = 60):
    """
    Stream-download unpacking remote ZIP content via HTTP Range requests.
    Skips already-extracted files (filling the bar by the compressed size).
    Works well with large files on Zenodo not hummering the server.

    A range slower than min_rate (bytes per second over min_rate_window seconds, a size string such as '100KB' or
    a number), or failing in any other way a retry can change, is requested again, up to retries attempts with
    timeout_second seconds between them (HTTPRangeReader).
    """
    if target_dir is None:
        target_dir = os.path.basename(url).split('.')[0]
    else:
        target_dir = os.path.expanduser(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    raw = HTTPRangeReader(url, retries=retries, timeout_second=timeout_second,
                          min_rate=min_rate, min_rate_window=min_rate_window)
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
    """A resource that does not exist, as the server states or as the delivered product shows (an archive without
    a file it must hold), which a retry cannot change (final())."""


class ProxyStatusError(requests.exceptions.ProxyError):
    """A proxy answered the request for an HTTPS tunnel (CONNECT) with an HTTP error status: status_line is its
    answer, such as '407 Proxy Authentication Required'."""

    def __init__(self, *args, status_line=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.status_line = status_line


# The HTTP error statuses that a retry cannot change: a bad request, a missing or refused authorization (407: of a
# proxy), no such resource, a method, state or content that the server does not accept. 408 (request timeout), 429
# (too many requests) and the 5xx server errors are temporary and retried, as every other status is.
FINAL_STATUS = (400, 401, 403, 404, 405, 407, 409, 410, 422)
# the start of the OSError text that http.client raises when a proxy answers the HTTPS tunnel request (CONNECT) with
# an HTTP status, followed by the status line of the proxy; requests wraps that OSError into its ProxyError
_TUNNEL_FAILED = 'Tunnel connection failed: '
# the longest message of an answer that is shown, in characters, and the most bytes of a body read for it
MESSAGE_LENGTH = 200
_MESSAGE_BYTES = 65536
# the fields of a JSON answer that hold its message, in the order they are taken, as the servers send them: message
# (the CDSE catalogue in its 'detail' object, the ASF burst service), msg (the items of the 'detail' list of the CDSE
# catalogue), detail (the CDSE catalogue), report (the ASF catalogue in its 'error' object), error_description and
# error (the CDSE token service)
_MESSAGE_FIELDS = ('message', 'msg', 'detail', 'report', 'error_description', 'error')


def status_line(response) -> str:
    """The status line that a server or proxy returned, such as '503 Service Unavailable'."""
    reason = response.reason
    if isinstance(reason, bytes):
        try:
            reason = reason.decode('utf-8')
        except UnicodeDecodeError:
            reason = reason.decode('iso-8859-1')
    return f'{response.status_code} {reason or ""}'.strip()


def _json_message(value):
    """The message of a parsed JSON answer: a string, the first of _MESSAGE_FIELDS of an object that holds one, or
    the messages of a list joined by '; '; None when it holds none."""
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        for key in _MESSAGE_FIELDS:
            text = _json_message(value.get(key))
            if text:
                return text
        return None
    if isinstance(value, list):
        return '; '.join(filter(None, map(_json_message, value))) or None
    return None


def message(response):
    """
    The error message that a server or a proxy returned in the body of its answer, whitespace collapsed and cut to
    MESSAGE_LENGTH characters: the message field of a JSON body (_MESSAGE_FIELDS), or the text of a plain-text
    body.

    Returns
    -------
    str or None
        The message; None when the answer carries none: an HTML or XML page, a JSON body without a message field,
        an empty or binary body, or a body that cannot be read.
    """
    import json
    body = b''
    try:
        for chunk in response.iter_content(8192):
            body += chunk
            if len(body) >= _MESSAGE_BYTES:
                break
    except (requests.RequestException, OSError):
        return None
    try:
        text = body.decode('utf-8-sig').strip()
    except UnicodeDecodeError:
        return None
    if not text:
        return None
    try:
        data = json.loads(text) if 'json' in response.headers.get('Content-Type', '').lower() or text[0] in '{[' \
            else text
    except ValueError:
        # not JSON: the text itself
        data = text
    if data is not text:
        text = _json_message(data)
        if text is None:
            return None
    elif text.startswith('<'):
        # an HTML or XML page
        return None
    text = ' '.join(text.split())
    if not text.isprintable():
        return None
    return text if len(text) <= MESSAGE_LENGTH else text[:MESSAGE_LENGTH - 3].rstrip() + '...'


def reply(response) -> str:
    """What a server or a proxy returned for a failed request: the message of its answer (message()), or its status
    line (status_line()) when the answer carries none."""
    return message(response) or status_line(response)


def _tunnel_status(error):
    """The status line that a proxy answered the HTTPS tunnel request (CONNECT) with, taken whole from the OSError
    of http.client inside the requests ProxyError, such as "501 Unsupported method ('CONNECT')"; None for a proxy
    that is not reached."""
    seen, todo = set(), [error]
    while todo:
        e = todo.pop()
        if not isinstance(e, BaseException) or id(e) in seen:
            continue
        seen.add(id(e))
        text = str(e)
        if isinstance(e, OSError) and text.startswith(_TUNNEL_FAILED):
            return text[len(_TUNNEL_FAILED):].strip() or None
        todo += [arg for arg in e.args if isinstance(arg, BaseException)]
        todo += [getattr(e, 'reason', None), e.__cause__, e.__context__]
    return None


def answer(error):
    """
    The status line that a server or a proxy returned for a failed request, such as '503 Service Unavailable': of
    an HTTP error status (requests.HTTPError with its response), or of a proxy that answered the HTTPS tunnel
    request with an HTTP status (ProxyStatusError, or the requests ProxyError it is made from).

    Returns
    -------
    str or None
        The status line; None when nothing was returned (no connection, DNS failure, timeout) and for every other
        error.
    """
    if isinstance(error, ProxyStatusError):
        return error.status_line
    if isinstance(error, requests.exceptions.ProxyError):
        return _tunnel_status(error)
    if isinstance(error, requests.HTTPError) and error.response is not None:
        return status_line(error.response)
    return None


def returned(error):
    """
    What a server or a proxy returned for a failed request, as its error shows it: the message of the answer or its
    status line (reply()) of an HTTP error status raised by http_error(), and the status line of a proxy that
    answered the HTTPS tunnel request with an HTTP status (ProxyStatusError).

    Returns
    -------
    str or None
        None when nothing was returned (no connection, DNS failure, timeout) and for every other error.
    """
    if isinstance(error, ProxyStatusError):
        return error.status_line
    if isinstance(error, requests.HTTPError):
        return getattr(error, 'reply', None)
    return None


def http_error(response, what):
    """
    requests.HTTPError of the response, with the message '<what>: <reply>', where reply is the message the server
    or proxy returned in its answer, or its status line when it returned none (reply()), also kept as the attribute
    reply of the error. The response is closed: a streamed one would hold its connection.
    """
    text = reply(response)
    response.close()
    error = requests.HTTPError(f'{what}: {text}', response=response)
    error.reply = text
    return error


def raise_for_status(response, what):
    """Raise http_error(response, what) for an HTTP error status (400 to 599)."""
    if 400 <= response.status_code < 600:
        raise http_error(response, what)


def send(method, url, *args, what=None, check=True, **kwargs):
    """
    method(url, *args, **kwargs), such as requests.get or session.head, with a failed answer raised as the
    message '<what>: <reply>' (what: the URL by default), where reply is the message the server or proxy returned
    (reply()), such as 'https://s1-cache-cdse.insar.dev/<uuid>: CDSE error: 404 Not Found ...': an HTTP error
    status (400 to 599) raises requests.HTTPError with the response, closed (check=False returns the response
    instead), and a proxy that answers the HTTPS tunnel request (CONNECT) with an HTTP status raises
    ProxyStatusError with its status line. A failure with no answer (no connection, DNS failure, timeout) raises as
    requests raises it.
    """
    try:
        response = method(url, *args, **kwargs)
    except requests.exceptions.ProxyError as e:
        line = answer(e)
        if line is None:
            raise
        raise ProxyStatusError(f'{what or url}: {line}', status_line=line) from None
    if check:
        raise_for_status(response, what or url)
    return response


def final(error) -> bool:
    """
    Whether a failed request fails the same way on every retry, so that the downloaders of the toolkit raise it
    at once instead of retrying it.

    Final: an HTTP error status of FINAL_STATUS, also from a proxy that answers the HTTPS tunnel request with it
    (answer()), a URL that cannot be requested (no scheme, a scheme with no connection adapter, an invalid URL), a
    redirect loop, and NotFound (a resource that does not exist, such as an orbit index that does not list the
    orbit file). Everything else is temporary and retried: a server or proxy that is not reached (connection error,
    DNS failure, timeout), HTTP 408, 429 and 5xx (also from a proxy), and a body that cannot be used (it may be a
    truncated transfer).
    """
    if isinstance(error, (requests.HTTPError, requests.exceptions.ProxyError)):
        line = answer(error)
        return line is not None and int(line.split()[0]) in FINAL_STATUS
    return isinstance(error, (NotFound, requests.exceptions.MissingSchema, requests.exceptions.InvalidSchema,
                              requests.exceptions.InvalidURL, requests.exceptions.TooManyRedirects))


def attempts(retries: int) -> int:
    """
    The number of attempts of a request for the retries argument of the toolkit's downloaders: retries counts the
    attempts, the first one included, and 0 makes one attempt, as 1 does.

    Raises
    ------
    ValueError
        For a negative retries.
    """
    if retries < 0:
        raise ValueError(f'retries must be 0 or more, got {retries}')
    return max(retries, 1)


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


def iter_body(response, min_rate='100KB', min_rate_window: float = 60, progress=None):
    """
    The body of a response requested with stream=True, chunk by chunk as the network delivers it, cut when the
    transfer stays slower than min_rate. One network read (up to 64 KB) is held at a time, so a body of any size
    can be written as it arrives (download(), HTTPRangeReader); read_body() collects it for a body that is needed in
    memory anyway.

    A read timeout alone never fires while bytes keep arriving, however slowly, so a trickling transfer is cut by the
    rate check instead, made after every network read; the caller retries it, and closing the response drops the
    connection, so the next attempt starts a new one. The rate is measured from the first byte of the body: a
    server that stays silent before it, as a proxy does while it fetches the data upstream, is bounded by the read
    timeout of the request. A transfer that keeps the minimum rate is never cut, however long it takes.

    Parameters
    ----------
    response : requests.Response
        A response requested with stream=True, its body not read yet.
    min_rate : str or float, optional
        Minimum transfer rate in bytes per second, a size string such as '100KB' or a number, checked over each
        min_rate_window. Default '100KB'.
    min_rate_window : float, optional
        Seconds over which the rate is measured. Default 60.
    progress : callable, optional
        Called with the byte count of every network read, such as the update of a progress bar.

    Yields
    ------
    bytes
        The next part of the body, transparently decoded.

    Raises
    ------
    TimeoutError
        The transfer stays slower than min_rate.
    requests.RequestException
        A network read that fails, raised as requests raises it when it reads a body itself (Response.iter_content):
        a read timeout as requests.ConnectionError, a broken transfer, also a body that ends before its
        Content-Length, as requests.ChunkedEncodingError.
    IOError
        A body shorter than its Content-Length that the read did not report.
    """
    import time
    from urllib3.exceptions import DecodeError, ProtocolError, ReadTimeoutError, SSLError

    if isinstance(min_rate, str):
        from dask.utils import parse_bytes
        min_rate = parse_bytes(min_rate)
    size = 0
    window_t = window_n = None
    # read1() returns what one network read brings, up to 64 KB, so the rate check never waits for
    # more; urllib3 before 2.0 has no read1() and its read() waits for the whole 64 KB
    read1 = getattr(response.raw, 'read1', None)
    while True:
        try:
            chunk = read1(64 * 1024, decode_content=True) if read1 else response.raw.read(64 * 1024, decode_content=True)
        except ProtocolError as e:
            raise requests.exceptions.ChunkedEncodingError(e)
        except DecodeError as e:
            raise requests.exceptions.ContentDecodingError(e)
        except ReadTimeoutError as e:
            raise requests.exceptions.ConnectionError(e)
        except SSLError as e:
            raise requests.exceptions.SSLError(e)
        if not chunk:
            break
        size += len(chunk)
        yield chunk
        if progress is not None:
            progress(len(chunk))
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
    if expected is not None and encoding == 'identity' and size != int(expected):
        raise IOError(f'truncated body: {size} of {expected} bytes')


def read_body(response, min_rate='100KB', min_rate_window: float = 60, progress=None) -> bytes:
    """
    The whole body of a response requested with stream=True, read by iter_body(): cut when the transfer stays
    slower than min_rate. For a body needed in memory anyway, such as a burst or a NISAR block; a body that is
    stored as a file is written from iter_body() as it arrives.

    Parameters and Raises as iter_body().

    Returns
    -------
    bytes
        The response body, transparently decoded.
    """
    body = bytearray()
    for chunk in iter_body(response, min_rate, min_rate_window, progress):
        body += chunk
    return bytes(body)


def fetch(url: str, session=None, headers=None, magic=None, retries: int = 30, timeout_second: float = 3,
          timeout=(10, 300), min_rate='100KB', min_rate_window: float = 60,
          debug: bool = False) -> bytes:
    """
    Download a whole response body into memory.

    Every failure that a retry can change is retried: connection errors, the HTTP errors other than FINAL_STATUS,
    a body shorter than its Content-Length, a body that is not the expected payload, and a transfer that stays
    slower than the minimum rate. The body is read by read_body(): a read timeout
    alone never fires while bytes keep arriving, however slowly, so a trickling transfer is cut by the rate check
    instead, made after every network read, and the next attempt starts a new connection. The rate is measured
    from the first byte of the body: a server that stays silent before it, as a proxy does while it fetches the
    data upstream, is bounded by the read timeout. A transfer that keeps the minimum rate is never cut, however
    long it takes.

    A failure that a retry cannot change (final()) raises at its first attempt. An explicit "does not exist" raises
    NotFound, with nothing printed, as the caller decides whether a missing resource is an error: HTTP 404 or 410,
    and an HTML page served with HTTP 200 in place of the expected payload (a soft 404 page). The other statuses of
    FINAL_STATUS (400, 401, 403, 405, 407, 409, 422), also from a proxy, a URL that cannot be requested and a
    redirect loop raise their error, printed as not retried. An HTTP error status, of the server or of a proxy,
    raises as '<url>: <reply>' (send()), where reply is the message the server or proxy returned, or its status
    line when it returned none, and its printed line gives the reply alone.

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
        Number of attempts, the first one included; 0 makes one attempt, as 1 does (attempts()). Default 30.
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
        Print the retried failed attempts too; the last failed attempt and one that is not retried are always
        printed, and NotFound never is. Default False.

    Returns
    -------
    bytes
        The response body.

    Raises
    ------
    NotFound
        The resource does not exist.
    ValueError
        For a negative retries, before any request.
    Exception
        A failure that a retry cannot change, at once, or the last error once all attempts have failed.
    """
    import time

    n = attempts(retries)
    if isinstance(min_rate, str):
        from dask.utils import parse_bytes
        min_rate = parse_bytes(min_rate)
    session = session or _session()
    for attempt in range(n):
        try:
            with send(session.get, url, headers=headers, stream=True, timeout=timeout, check=False) as response:
                if response.status_code in (404, 410):
                    raise NotFound(f'{url}: {reply(response)}')
                raise_for_status(response, url)
                body = read_body(response, min_rate, min_rate_window)
                if magic is not None and not body[:8].startswith(tuple(magic)):
                    if 'text/html' in response.headers.get('Content-Type', '').lower():
                        raise NotFound(f'HTML page instead of the expected payload: {url}')
                    raise IOError(f'unexpected payload {body[:8]!r}')
                return body
        except NotFound:
            # an expected answer for some callers (an offshore DEM tile, a map tile the service does not have):
            # the caller prints its own message, or raises
            raise
        except Exception as e:
            stop = final(e)
            if debug or stop or attempt + 1 == n:
                print(f'ERROR: download attempt {attempt + 1}/{n} failed{" (not retried)" if stop else ""} '
                      f'for {url}: {returned(e) or e}')
            if stop or attempt + 1 == n:
                raise
            time.sleep(timeout_second)


def download(url: str, dst: str, retries: int = 30, timeout_second: float = 3, timeout=(10, 300),
             min_rate='100KB', min_rate_window: float = 60):
    """
    Download url into the file dst, written as the body arrives (iter_body), so that the memory used is one
    network read whatever the size of the file.

    A dst whose size reaches the Content-Length is kept; an incomplete one is removed and downloaded again. Every
    failure that a retry can change, also a transfer slower than min_rate, restarts the download from its first
    byte on a new connection, up to retries attempts; a failure that a retry cannot change (final()) raises at once.

    Parameters
    ----------
    url : str
        URL to download.
    dst : str
        The file to write.
    retries : int, optional
        Number of attempts, the first one included; 0 makes one attempt, as 1 does (attempts()). Default 30.
    timeout_second : float, optional
        Seconds between attempts. Default 3.
    timeout : tuple, optional
        (connect, read) timeouts in seconds. Default (10, 300).
    min_rate : str or float, optional
        Minimum transfer rate in bytes per second, a size string such as '100KB' or a number, checked over each
        min_rate_window (iter_body). Default '100KB'.
    min_rate_window : float, optional
        Seconds over which the rate is measured. Default 60.
    """
    import os
    import time
    import requests
    from tqdm.auto import tqdm

    downloaded = os.path.getsize(dst) if os.path.exists(dst) else 0
    head = send(requests.head, url, allow_redirects=True, timeout=timeout)
    expected = int(head.headers.get("content-length", 0))
    #print ('downloaded', downloaded, 'expected', expected)
    if downloaded and expected and downloaded >= expected:
        print(f"{dst} is already fully downloaded.")
        return
    if os.path.exists(dst):
        print(f'{dst} is incompletely downloaded, removing it.')
        os.remove(dst)
    n = attempts(retries)
    for attempt in range(n):
        try:
            with send(requests.get, url, stream=True, timeout=timeout) as resp:
                total = int(resp.headers.get('content-length', 0))
                with open(dst, 'wb') as f, \
                     tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
                          desc=os.path.basename(dst)) as bar:
                    for chunk in iter_body(resp, min_rate, min_rate_window, progress=bar.update):
                        f.write(chunk)
            return
        except Exception as e:
            stop = final(e)
            print(f'ERROR: download attempt {attempt + 1}/{n} failed{" (not retried)" if stop else ""} '
                  f'for {url}: {returned(e) or e}')
            if stop or attempt + 1 == n:
                raise
            time.sleep(timeout_second)
