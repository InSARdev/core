# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2026, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
"""
Pure Python solid Earth tides computation.

Vectorized numpy translation of GMTSAR's solid_tide.c (Dennis Milbert's algorithm).
Implements IERS 2003 conventions for degree-2/3 tidal displacement with
frequency-dependent and out-of-phase corrections.
"""
import numpy as np

# WGS84 constants
_A = 6378137.0
_E2 = 6.69438002290341574957e-3

_PI = np.pi
_PI2 = 2.0 * np.pi
_RAD = 180.0 / np.pi  # degrees per radian
_DEG2RAD = np.pi / 180.0


def solid_tide(lons, lats, dt):
    """Compute solid Earth tide displacement at given locations and UTC time.

    Parameters
    ----------
    lons : array_like
        Longitude in degrees (any shape).
    lats : array_like
        Latitude in degrees (same shape as lons).
    dt : datetime.datetime
        UTC time.

    Returns
    -------
    east, north, up : np.ndarray
        Displacement in meters, same shape as input.
    """
    lons = np.asarray(lons, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)
    orig_shape = lons.shape

    # Flatten for vectorized processing
    lon_flat = lons.ravel()
    lat_flat = lats.ravel()

    # Convert UTC datetime to MJD
    yr, mo, dy = dt.year, dt.month, dt.day
    hr, mn, sec = dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6
    mjd, fmjd = _civmjd(yr, mo, dy, hr, mn, sec)

    # Station ECEF coordinates (N, 3)
    gla = lat_flat / _RAD
    glo = lon_flat / _RAD
    xsta = _geo2xyz_vec(gla, glo, 0.0)  # (N, 3)

    # Sun and Moon ECEF (scalar, same for all points)
    rsun = _sunxyz(mjd, fmjd)   # (3,)
    rmoon = _moonxyz(mjd, fmjd)  # (3,)

    # Tidal displacement in ECEF (N, 3)
    dxtide = _detide(xsta, mjd, fmjd, rsun, rmoon)

    # Convert ECEF displacement to ENU (north, east, up)
    north, east, up = _rge_vec(gla, glo, dxtide)

    return east.reshape(orig_shape), north.reshape(orig_shape), up.reshape(orig_shape)


# ---------------------------------------------------------------------------
# Time conversion helpers (scalar)
# ---------------------------------------------------------------------------

def _civmjd(iyr, imo, idy, ihr, imn, sec):
    """Convert civil date to Modified Julian Date."""
    if imo <= 2:
        y = iyr - 1
        m = imo + 12
    else:
        y = iyr
        m = imo
    it1 = int(365.25 * y)
    it2 = int(30.6001 * (m + 1))
    mjd = float(it1 + it2 + idy - 679019)
    fmjd = (3600.0 * ihr + 60.0 * imn + sec) / 86400.0
    return mjd, fmjd


def _utc2tt(tsec, mjd0):
    """Convert UTC seconds to Terrestrial Time seconds.

    UTC → TAI (add leap seconds) → TT (add 32.184s).
    """
    # Resolve MJD for leap second lookup
    ttsec = tsec
    mjd0t = mjd0
    while ttsec >= 86400.0:
        ttsec -= 86400.0
        mjd0t += 1
    while ttsec < 0.0:
        ttsec += 86400.0
        mjd0t -= 1
    tai_utc = _gpsleap(mjd0t) + 19.0  # _gpsleap returns TAI-UTC-19
    return tsec + tai_utc + 32.184


def _gpsleap(mjd0t):
    """Return GPS leap seconds for given MJD."""
    # TAI-UTC table (IERS Bulletin C)
    if mjd0t >= 57754:
        tai_utc = 37.0
    elif mjd0t >= 57204:
        tai_utc = 36.0
    elif mjd0t >= 56109:
        tai_utc = 35.0
    elif mjd0t >= 54832:
        tai_utc = 34.0
    elif mjd0t >= 53736:
        tai_utc = 33.0
    elif mjd0t >= 51179:
        tai_utc = 32.0
    elif mjd0t >= 50630:
        tai_utc = 31.0
    elif mjd0t >= 50083:
        tai_utc = 30.0
    elif mjd0t >= 49534:
        tai_utc = 29.0
    elif mjd0t >= 49169:
        tai_utc = 28.0
    elif mjd0t >= 48804:
        tai_utc = 27.0
    elif mjd0t >= 48257:
        tai_utc = 26.0
    elif mjd0t >= 47892:
        tai_utc = 25.0
    elif mjd0t >= 47161:
        tai_utc = 24.0
    elif mjd0t >= 46247:
        tai_utc = 23.0
    elif mjd0t >= 45516:
        tai_utc = 22.0
    elif mjd0t >= 45151:
        tai_utc = 21.0
    elif mjd0t >= 44786:
        tai_utc = 20.0
    elif mjd0t >= 44239:
        tai_utc = 19.0
    else:
        tai_utc = 19.0
    return tai_utc - 19.0


def _getghar(mjd, fmjd):
    """Compute Greenwich Hour Angle in radians. Input is UTC."""
    d = (round(mjd) - 51544) + (fmjd - 0.5)
    ghad = 280.460618375040 + 360.98564736628620 * d
    i = int(ghad / 360.0)
    ghar = (ghad - i * 360.0) / _RAD
    while ghar > _PI2:
        ghar -= _PI2
    while ghar < 0.0:
        ghar += _PI2
    return ghar


# ---------------------------------------------------------------------------
# Coordinate conversions
# ---------------------------------------------------------------------------

def _geo2xyz_vec(gla, glo, eht):
    """Geodetic (lat_rad, lon_rad, height) to ECEF. Vectorized over N points.

    Returns (N, 3) array.
    """
    sla = np.sin(gla)
    cla = np.cos(gla)
    w2 = 1.0 - _E2 * sla * sla
    w = np.sqrt(w2)
    en = _A / w
    x = (en + eht) * cla * np.cos(glo)
    y = (en + eht) * cla * np.sin(glo)
    z = (en * (1.0 - _E2) + eht) * sla
    return np.column_stack([x, y, z])


def _rge_vec(gla, glo, dxyz):
    """ECEF displacement (N,3) to ENU (north, east, up). Vectorized."""
    sb = np.sin(gla)
    cb = np.cos(gla)
    sl = np.sin(glo)
    cl = np.cos(glo)
    dx = dxyz[:, 0]
    dy = dxyz[:, 1]
    dz = dxyz[:, 2]
    north = -sb * cl * dx - sb * sl * dy + cb * dz
    east = -sl * dx + cl * dy
    up = cb * cl * dx + cb * sl * dy + sb * dz
    return north, east, up


# ---------------------------------------------------------------------------
# Sun and Moon ephemeris (scalar — computed once per datetime)
# ---------------------------------------------------------------------------

def _rot1(theta, x, y, z):
    """Rotate about axis 1."""
    s, c = np.sin(theta), np.cos(theta)
    return np.array([x, c * y + s * z, c * z - s * y])


def _rot3(theta, x, y, z):
    """Rotate about axis 3."""
    s, c = np.sin(theta), np.cos(theta)
    return np.array([c * x + s * y, c * y - s * x, z])


def _sunxyz(mjd, fmjd):
    """Low-precision geocentric Sun position in ECEF (meters)."""
    obe = 23.43929111 / _RAD
    sobe = np.sin(obe)
    cobe = np.cos(obe)
    opod = 282.9400

    tsecutc = fmjd * 86400.0
    tsectt = _utc2tt(tsecutc, mjd)
    fmjdtt = tsectt / 86400.0

    tjdtt = round(mjd) + fmjdtt + 2400000.5
    t = (tjdtt - 2451545.0) / 36525.0
    emdeg = 357.5256 + 35999.049 * t
    em = emdeg / _RAD
    em2 = em + em

    r = (149.619 - 2.499 * np.cos(em) - 0.021 * np.cos(em2)) * 1.0e9
    slond = opod + emdeg + (6892.0 * np.sin(em) + 72.0 * np.sin(em2)) / 3600.0
    slond = slond + 1.39720 * t

    slon = slond / _RAD
    sslon = np.sin(slon)
    cslon = np.cos(slon)

    rs1 = r * cslon
    rs2 = r * sslon * cobe
    rs3 = r * sslon * sobe

    ghar = _getghar(mjd, fmjd)
    rs = _rot3(ghar, rs1, rs2, rs3)
    return rs


def _moonxyz(mjd, fmjd):
    """Low-precision geocentric Moon position in ECEF (meters)."""
    tsecutc = fmjd * 86400.0
    tsectt = _utc2tt(tsecutc, mjd)
    fmjdtt = tsectt / 86400.0

    tjdtt = round(mjd) + fmjdtt + 2400000.5
    t = (tjdtt - 2451545.0) / 36525.0

    el0 = 218.316170 + (481267.880880 - 1.3972) * t
    el = 134.962920 + 477198.867530 * t
    elp = 357.525430 + 35999.049440 * t
    f = 93.272830 + 483202.018730 * t
    d = 297.850270 + 445267.111350 * t

    selond = (el0
              + 22640.0 / 3600.0 * np.sin(el / _RAD)
              + 769.0 / 3600.0 * np.sin(2 * el / _RAD)
              - 4586.0 / 3600.0 * np.sin((el - 2 * d) / _RAD)
              + 2370.0 / 3600.0 * np.sin(2 * d / _RAD)
              - 668.0 / 3600.0 * np.sin(elp / _RAD)
              - 412.0 / 3600.0 * np.sin(2 * f / _RAD)
              - 212.0 / 3600.0 * np.sin((2 * el - 2 * d) / _RAD)
              - 206.0 / 3600.0 * np.sin((el + elp - 2 * d) / _RAD)
              + 192.0 / 3600.0 * np.sin((el + 2 * d) / _RAD)
              - 165.0 / 3600.0 * np.sin((elp - 2 * d) / _RAD)
              + 148.0 / 3600.0 * np.sin((el - elp) / _RAD)
              - 125.0 / 3600.0 * np.sin(d / _RAD)
              - 110.0 / 3600.0 * np.sin((el + elp) / _RAD)
              - 55.0 / 3600.0 * np.sin((2 * f - 2 * d) / _RAD))

    q = 412.0 / 3600.0 * np.sin(2 * f / _RAD) + 541.0 / 3600.0 * np.sin(elp / _RAD)

    selatd = (+18520.0 / 3600.0 * np.sin((f + selond - el0 + q) / _RAD)
              - 526.0 / 3600.0 * np.sin((f - 2 * d) / _RAD)
              + 44.0 / 3600.0 * np.sin((el + f - 2 * d) / _RAD)
              - 31.0 / 3600.0 * np.sin((-el + f - 2 * d) / _RAD)
              - 25.0 / 3600.0 * np.sin((-2 * el + f) / _RAD)
              - 23.0 / 3600.0 * np.sin((elp + f - 2 * d) / _RAD)
              + 21.0 / 3600.0 * np.sin((-el + f) / _RAD)
              + 11.0 / 3600.0 * np.sin((-elp + f - 2 * d) / _RAD))

    rse = (385000.0 * 1000.0
           - 20905.0 * 1000.0 * np.cos(el / _RAD)
           - 3699.0 * 1000.0 * np.cos((2 * d - el) / _RAD)
           - 2956.0 * 1000.0 * np.cos(2 * d / _RAD)
           - 570.0 * 1000.0 * np.cos(2 * el / _RAD)
           + 246.0 * 1000.0 * np.cos((2 * el - 2 * d) / _RAD)
           - 205.0 * 1000.0 * np.cos((elp - 2 * d) / _RAD)
           - 171.0 * 1000.0 * np.cos((el + 2 * d) / _RAD)
           - 152.0 * 1000.0 * np.cos((el + elp - 2 * d) / _RAD))

    selond = selond + 1.39720 * t
    oblir = 23.439291110 / _RAD

    sselat = np.sin(selatd / _RAD)
    cselat = np.cos(selatd / _RAD)
    sselon = np.sin(selond / _RAD)
    cselon = np.cos(selond / _RAD)

    t1 = rse * cselon * cselat
    t2 = rse * sselon * cselat
    t3 = rse * sselat

    rm = _rot1(-oblir, t1, t2, t3)
    ghar = _getghar(mjd, fmjd)
    rm = _rot3(ghar, rm[0], rm[1], rm[2])
    return rm


# ---------------------------------------------------------------------------
# Main tidal displacement (vectorized over N stations)
# ---------------------------------------------------------------------------

def _detide(xsta, mjd, fmjd, xsun, xmon):
    """Compute tidal displacement in ECEF for N stations.

    Parameters
    ----------
    xsta : (N, 3) array — station ECEF coordinates
    mjd, fmjd : float — Modified Julian Date
    xsun, xmon : (3,) arrays — Sun/Moon ECEF coordinates

    Returns
    -------
    dxtide : (N, 3) array — displacement in ECEF (meters)
    """
    # Love numbers
    h20 = 0.6078
    l20 = 0.0847
    h3 = 0.292
    l3 = 0.015

    # Time conversions for step2
    tsecutc = fmjd * 86400.0
    tsectt = _utc2tt(tsecutc, mjd)
    fmjdtt = tsectt / 86400.0
    dmjdtt = mjd + fmjdtt
    t = (dmjdtt - 51544.0) / 36525.0
    fhr = (dmjdtt - np.floor(dmjdtt)) * 24.0

    # Station norms and dot products with sun/moon
    # xsta: (N,3), xsun: (3,), xmon: (3,)
    rsta = np.sqrt(np.sum(xsta ** 2, axis=1))  # (N,)
    rsun = np.sqrt(np.sum(xsun ** 2))
    rmon = np.sqrt(np.sum(xmon ** 2))

    scs = np.dot(xsta, xsun)  # (N,)
    scm = np.dot(xsta, xmon)  # (N,)

    scsun = scs / rsta / rsun  # (N,)
    scmon = scm / rsta / rmon  # (N,)

    # Latitude-dependent h2, l2
    cosphi = np.sqrt(xsta[:, 0] ** 2 + xsta[:, 1] ** 2) / rsta
    h2 = h20 - 0.0006 * (1.0 - 1.5 * cosphi ** 2)
    l2 = l20 + 0.0002 * (1.0 - 1.5 * cosphi ** 2)

    # Legendre terms
    p2sun = 3.0 * (h2 / 2.0 - l2) * scsun ** 2 - h2 / 2.0
    p2mon = 3.0 * (h2 / 2.0 - l2) * scmon ** 2 - h2 / 2.0
    p3sun = 2.5 * (h3 - 3.0 * l3) * scsun ** 3 + 1.5 * (l3 - h3) * scsun
    p3mon = 2.5 * (h3 - 3.0 * l3) * scmon ** 3 + 1.5 * (l3 - h3) * scmon

    x2sun = 3.0 * l2 * scsun
    x2mon = 3.0 * l2 * scmon
    x3sun = 1.5 * l3 * (5.0 * scsun ** 2 - 1.0)
    x3mon = 1.5 * l3 * (5.0 * scmon ** 2 - 1.0)

    # Mass ratios and scale factors
    mass_ratio_sun = 332945.943062
    mass_ratio_moon = 0.012300034
    re = 6378136.55
    fac2sun = mass_ratio_sun * re * (re / rsun) ** 3
    fac2mon = mass_ratio_moon * re * (re / rmon) ** 3
    fac3sun = fac2sun * (re / rsun)
    fac3mon = fac2mon * (re / rmon)

    # Total displacement (N, 3)
    # xsun_n = xsun[np.newaxis, :]  (1,3), xsta/rsta[:, np.newaxis] = (N,1)
    xsun_n = xsun[np.newaxis, :]  # (1, 3)
    xmon_n = xmon[np.newaxis, :]  # (1, 3)
    xsta_n = xsta / rsta[:, np.newaxis]  # (N, 3) normalized

    dxtide = (fac2sun * (x2sun[:, np.newaxis] * xsun_n / rsun + p2sun[:, np.newaxis] * xsta_n)
              + fac2mon * (x2mon[:, np.newaxis] * xmon_n / rmon + p2mon[:, np.newaxis] * xsta_n)
              + fac3sun * (x3sun[:, np.newaxis] * xsun_n / rsun + p3sun[:, np.newaxis] * xsta_n)
              + fac3mon * (x3mon[:, np.newaxis] * xmon_n / rmon + p3mon[:, np.newaxis] * xsta_n))

    # Out-of-phase corrections
    dxtide += _st1idiu_vec(xsta, xsun, xmon, fac2sun, fac2mon)
    dxtide += _st1isem_vec(xsta, xsun, xmon, fac2sun, fac2mon)
    dxtide += _st1l1_vec(xsta, xsun, xmon, fac2sun, fac2mon)

    # Step 2 frequency-dependent corrections
    dxtide += _step2diu_vec(xsta, fhr, t)
    dxtide += _step2lon_vec(xsta, fhr, t)

    return dxtide


# ---------------------------------------------------------------------------
# Out-of-phase corrections (vectorized)
# ---------------------------------------------------------------------------

def _st1idiu_vec(xsta, xsun, xmon, fac2sun, fac2mon):
    """Out-of-phase diurnal correction. (N,3)"""
    dhi = -0.0025
    dli = -0.0007

    rsta = np.sqrt(np.sum(xsta ** 2, axis=1))
    sinphi = xsta[:, 2] / rsta
    cosphi = np.sqrt(xsta[:, 0] ** 2 + xsta[:, 1] ** 2) / rsta
    cos2phi = cosphi ** 2 - sinphi ** 2
    sinla = xsta[:, 1] / cosphi / rsta
    cosla = xsta[:, 0] / cosphi / rsta
    rmon = np.sqrt(np.sum(xmon ** 2))
    rsun = np.sqrt(np.sum(xsun ** 2))

    # Sun terms
    drsun = -3.0 * dhi * sinphi * cosphi * fac2sun * xsun[2] * (xsun[0] * sinla - xsun[1] * cosla) / rsun ** 2
    dnsun = -3.0 * dli * cos2phi * fac2sun * xsun[2] * (xsun[0] * sinla - xsun[1] * cosla) / rsun ** 2
    desun = -3.0 * dli * sinphi * fac2sun * xsun[2] * (xsun[0] * cosla + xsun[1] * sinla) / rsun ** 2

    # Moon terms
    drmon = -3.0 * dhi * sinphi * cosphi * fac2mon * xmon[2] * (xmon[0] * sinla - xmon[1] * cosla) / rmon ** 2
    dnmon = -3.0 * dli * cos2phi * fac2mon * xmon[2] * (xmon[0] * sinla - xmon[1] * cosla) / rmon ** 2
    demon = -3.0 * dli * sinphi * fac2mon * xmon[2] * (xmon[0] * cosla + xmon[1] * sinla) / rmon ** 2

    dr = drsun + drmon
    dn = dnsun + dnmon
    de = desun + demon

    xcorsta = np.empty_like(xsta)
    xcorsta[:, 0] = dr * cosla * cosphi - de * sinla - dn * sinphi * cosla
    xcorsta[:, 1] = dr * sinla * cosphi + de * cosla - dn * sinphi * sinla
    xcorsta[:, 2] = dr * sinphi + dn * cosphi
    return xcorsta


def _st1isem_vec(xsta, xsun, xmon, fac2sun, fac2mon):
    """Out-of-phase semi-diurnal correction. (N,3)"""
    dhi = -0.0022
    dli = -0.0007

    rsta = np.sqrt(np.sum(xsta ** 2, axis=1))
    sinphi = xsta[:, 2] / rsta
    cosphi = np.sqrt(xsta[:, 0] ** 2 + xsta[:, 1] ** 2) / rsta
    sinla = xsta[:, 1] / cosphi / rsta
    cosla = xsta[:, 0] / cosphi / rsta
    costwola = cosla ** 2 - sinla ** 2
    sintwola = 2.0 * cosla * sinla
    rmon = np.sqrt(np.sum(xmon ** 2))
    rsun = np.sqrt(np.sum(xsun ** 2))

    sun_xx_yy = xsun[0] ** 2 - xsun[1] ** 2
    sun_xy = xsun[0] * xsun[1]
    mon_xx_yy = xmon[0] ** 2 - xmon[1] ** 2
    mon_xy = xmon[0] * xmon[1]

    drsun = -3.0 / 4.0 * dhi * cosphi ** 2 * fac2sun * (sun_xx_yy * sintwola - 2.0 * sun_xy * costwola) / rsun ** 2
    drmon = -3.0 / 4.0 * dhi * cosphi ** 2 * fac2mon * (mon_xx_yy * sintwola - 2.0 * mon_xy * costwola) / rmon ** 2
    dnsun = 1.5 * dli * sinphi * cosphi * fac2sun * (sun_xx_yy * sintwola - 2.0 * sun_xy * costwola) / rsun ** 2
    dnmon = 1.5 * dli * sinphi * cosphi * fac2mon * (mon_xx_yy * sintwola - 2.0 * mon_xy * costwola) / rmon ** 2
    desun = -1.5 * dli * cosphi * fac2sun * (sun_xx_yy * costwola + 2.0 * sun_xy * sintwola) / rsun ** 2
    demon = -1.5 * dli * cosphi * fac2mon * (mon_xx_yy * costwola + 2.0 * mon_xy * sintwola) / rmon ** 2

    dr = drsun + drmon
    dn = dnsun + dnmon
    de = desun + demon

    xcorsta = np.empty_like(xsta)
    xcorsta[:, 0] = dr * cosla * cosphi - de * sinla - dn * sinphi * cosla
    xcorsta[:, 1] = dr * sinla * cosphi + de * cosla - dn * sinphi * sinla
    xcorsta[:, 2] = dr * sinphi + dn * cosphi
    return xcorsta


def _st1l1_vec(xsta, xsun, xmon, fac2sun, fac2mon):
    """Latitude-dependent l^(1) correction. (N,3)"""
    l1d = 0.0012
    l1sd = 0.0024

    rsta = np.sqrt(np.sum(xsta ** 2, axis=1))
    sinphi = xsta[:, 2] / rsta
    cosphi = np.sqrt(xsta[:, 0] ** 2 + xsta[:, 1] ** 2) / rsta
    sinla = xsta[:, 1] / cosphi / rsta
    cosla = xsta[:, 0] / cosphi / rsta
    rmon = np.sqrt(np.sum(xmon ** 2))
    rsun = np.sqrt(np.sum(xsun ** 2))

    # Diurnal band
    l1 = l1d
    dnsun = -l1 * sinphi ** 2 * fac2sun * xsun[2] * (xsun[0] * cosla + xsun[1] * sinla) / rsun ** 2
    dnmon = -l1 * sinphi ** 2 * fac2mon * xmon[2] * (xmon[0] * cosla + xmon[1] * sinla) / rmon ** 2
    desun = l1 * sinphi * (cosphi ** 2 - sinphi ** 2) * fac2sun * xsun[2] * (xsun[0] * sinla - xsun[1] * cosla) / rsun ** 2
    demon = l1 * sinphi * (cosphi ** 2 - sinphi ** 2) * fac2mon * xmon[2] * (xmon[0] * sinla - xmon[1] * cosla) / rmon ** 2
    de = 3.0 * (desun + demon)
    dn = 3.0 * (dnsun + dnmon)

    xcorsta = np.empty_like(xsta)
    xcorsta[:, 0] = -de * sinla - dn * sinphi * cosla
    xcorsta[:, 1] = de * cosla - dn * sinphi * sinla
    xcorsta[:, 2] = dn * cosphi

    # Semi-diurnal band
    l1 = l1sd
    costwola = cosla ** 2 - sinla ** 2
    sintwola = 2.0 * cosla * sinla

    sun_xx_yy = xsun[0] ** 2 - xsun[1] ** 2
    sun_xy = xsun[0] * xsun[1]
    mon_xx_yy = xmon[0] ** 2 - xmon[1] ** 2
    mon_xy = xmon[0] * xmon[1]

    dnsun = -l1 / 2.0 * sinphi * cosphi * fac2sun * (sun_xx_yy * costwola + 2.0 * sun_xy * sintwola) / rsun ** 2
    dnmon = -l1 / 2.0 * sinphi * cosphi * fac2mon * (mon_xx_yy * costwola + 2.0 * mon_xy * sintwola) / rmon ** 2
    desun = -l1 / 2.0 * sinphi ** 2 * cosphi * fac2sun * (sun_xx_yy * sintwola - 2.0 * sun_xy * costwola) / rsun ** 2
    demon = -l1 / 2.0 * sinphi ** 2 * cosphi * fac2mon * (mon_xx_yy * sintwola - 2.0 * mon_xy * costwola) / rmon ** 2
    de = 3.0 * (desun + demon)
    dn = 3.0 * (dnsun + dnmon)

    xcorsta[:, 0] += -de * sinla - dn * sinphi * cosla
    xcorsta[:, 1] += de * cosla - dn * sinphi * sinla
    xcorsta[:, 2] += dn * cosphi

    return xcorsta


# ---------------------------------------------------------------------------
# Step 2 frequency-dependent corrections (vectorized)
# ---------------------------------------------------------------------------

# Table 7.5a — diurnal band (31 harmonics)
# Columns: s, h, p, N', ps, dR(ip), dR(op), dT(ip), dT(op)
_DATDI_DIU = np.array([
    [-3., 0., 2., 0., 0., -.01, -.01, 0., 0.],
    [-3., 2., 0., 0., 0., -.01, -.01, 0., 0.],
    [-2., 0., 1., -1., 0., -.02, -.01, 0., 0.],
    [-2., 0., 1., 0., 0., -.08, 0., .01, .01],
    [-2., 2., -1., 0., 0., -.02, -.01, 0., 0.],
    [-1., 0., 0., -1., 0., -.10, 0., 0., 0.],
    [-1., 0., 0., 0., 0., -.51, 0., -.02, .03],
    [-1., 2., 0., 0., 0., .01, 0., 0., 0.],
    [0., -2., 1., 0., 0., .01, 0., 0., 0.],
    [0., 0., -1., 0., 0., .02, .01, 0., 0.],
    [0., 0., 1., 0., 0., .06, 0., 0., 0.],
    [0., 0., 1., 1., 0., .01, 0., 0., 0.],
    [0., 2., -1., 0., 0., .01, 0., 0., 0.],
    [1., -3., 0., 0., 1., -.06, 0., 0., 0.],
    [1., -2., 0., 1., 0., .01, 0., 0., 0.],
    [1., -2., 0., 0., 0., -1.23, -.07, .06, .01],
    [1., -1., 0., 0., -1., .02, 0., 0., 0.],
    [1., -1., 0., 0., 1., .04, 0., 0., 0.],
    [1., 0., 0., -1., 0., -.22, .01, .01, 0.],
    [1., 0., 0., 0., 0., 12., -.78, -.67, -.03],
    [1., 0., 0., 1., 0., 1.73, -.12, -.10, 0.],
    [1., 0., 0., 2., 0., -.04, 0., 0., 0.],
    [1., 1., 0., 0., -1., -.50, -.01, .03, 0.],
    [1., 1., 0., 0., 1., .01, 0., 0., 0.],
    [1., 1., 0., 1., -1., -.01, 0., 0., 0.],
    [1., 2., -2., 0., 0., -.01, 0., 0., 0.],
    [1., 2., 0., 0., 0., -.11, .01, .01, 0.],
    [2., -2., 1., 0., 0., -.01, 0., 0., 0.],
    [2., 0., -1., 0., 0., -.02, .02, 0., .01],
    [3., 0., 0., 0., 0., 0., .01, 0., .01],
    [3., 0., 0., 1., 0., 0., .01, 0., 0.],
], dtype=np.float64)

# Table 7.5b — long-period band (5 harmonics)
_DATDI_LON = np.array([
    [0., 0., 0., 1., 0., .47, .23, .16, .07],
    [0., 2., 0., 0., 0., -.20, -.12, -.11, -.05],
    [1., 0., -1., 0., 0., -.11, -.08, -.09, -.04],
    [2., 0., 0., 0., 0., -.13, -.11, -.15, -.07],
    [2., 0., 0., 1., 0., -.05, -.05, -.06, -.03],
], dtype=np.float64)


def _step2diu_vec(xsta, fhr, t):
    """Frequency-dependent diurnal corrections. (N,3)"""
    s = 218.31664563 + 481267.88194 * t - 0.0014663889 * t ** 2 + 0.00000185139 * t ** 3
    tau = fhr * 15.0 + 280.46061840 + 36000.77005360 * t + 0.000387930 * t ** 2 - 0.0000000258 * t ** 3 - s
    pr = 1.396971278 * t + 0.000308889 * t ** 2 + 0.000000021 * t ** 3 + 0.000000007 * t ** 4
    s += pr
    h = 280.46645 + 36000.7697489 * t + 0.00030322222 * t ** 2 + 0.000000020 * t ** 3 - 0.00000000654 * t ** 4
    p = 83.35324312 + 4069.01363525 * t - 0.01032172222 * t ** 2 - 0.0000124991 * t ** 3 + 0.00000005263 * t ** 4
    zns = 234.95544499 + 1934.13626197 * t - 0.00207561111 * t ** 2 - 0.00000213944 * t ** 3 + 0.00000001650 * t ** 4
    ps = 282.93734098 + 1.71945766667 * t + 0.00045688889 * t ** 2 - 0.00000001778 * t ** 3 - 0.00000000334 * t ** 4

    s = s % 360.0
    tau = tau % 360.0
    h = h % 360.0
    p = p % 360.0
    zns = zns % 360.0
    ps = ps % 360.0

    rsta = np.sqrt(np.sum(xsta ** 2, axis=1))  # (N,)
    sinphi = xsta[:, 2] / rsta
    cosphi = np.sqrt(xsta[:, 0] ** 2 + xsta[:, 1] ** 2) / rsta
    cosla = xsta[:, 0] / cosphi / rsta
    sinla = xsta[:, 1] / cosphi / rsta
    zla = np.arctan2(xsta[:, 1], xsta[:, 0])  # (N,)

    xcorsta = np.zeros_like(xsta)  # (N, 3)

    dat = _DATDI_DIU
    # Compute all thetaf values at once: (31,) scalar angles
    thetaf = (tau + dat[:, 0] * s + dat[:, 1] * h + dat[:, 2] * p + dat[:, 3] * zns + dat[:, 4] * ps) * _DEG2RAD  # (31,)

    # For each harmonic, thetaf+zla is (31, N) via broadcasting
    thetaf_zla = thetaf[:, np.newaxis] + zla[np.newaxis, :]  # (31, N)

    sin_tf = np.sin(thetaf_zla)  # (31, N)
    cos_tf = np.cos(thetaf_zla)  # (31, N)

    # dr, dn, de for each harmonic: (31, N)
    dr = (dat[:, 5:6] * 2.0 * sinphi[np.newaxis, :] * cosphi[np.newaxis, :] * sin_tf
          + dat[:, 6:7] * 2.0 * sinphi[np.newaxis, :] * cosphi[np.newaxis, :] * cos_tf)  # broadcasting (31,1)*(1,N)
    # Fix: use proper column slicing
    dR_ip = dat[:, 5]  # (31,)
    dR_op = dat[:, 6]
    dT_ip = dat[:, 7]
    dT_op = dat[:, 8]

    dr = (dR_ip[:, np.newaxis] * 2.0 * sinphi * cosphi * sin_tf
          + dR_op[:, np.newaxis] * 2.0 * sinphi * cosphi * cos_tf)  # (31, N)

    cos2phi_sin2phi = cosphi ** 2 - sinphi ** 2  # (N,)
    dn = (dT_ip[:, np.newaxis] * cos2phi_sin2phi * sin_tf
          + dT_op[:, np.newaxis] * cos2phi_sin2phi * cos_tf)  # (31, N)

    # de uses dT_ip and dT_op (V.Dehant correction)
    de = (dT_ip[:, np.newaxis] * sinphi * cos_tf
          - dT_op[:, np.newaxis] * sinphi * sin_tf)  # (31, N)

    # Sum over harmonics: (N,)
    dr_tot = dr.sum(axis=0)
    dn_tot = dn.sum(axis=0)
    de_tot = de.sum(axis=0)

    xcorsta[:, 0] = dr_tot * cosla * cosphi - de_tot * sinla - dn_tot * sinphi * cosla
    xcorsta[:, 1] = dr_tot * sinla * cosphi + de_tot * cosla - dn_tot * sinphi * sinla
    xcorsta[:, 2] = dr_tot * sinphi + dn_tot * cosphi

    return xcorsta / 1000.0  # mm to meters


def _step2lon_vec(xsta, fhr, t):
    """Frequency-dependent long-period corrections. (N,3)"""
    s = 218.31664563 + 481267.88194 * t - 0.0014663889 * t ** 2 + 0.00000185139 * t ** 3
    pr = 1.396971278 * t + 0.000308889 * t ** 2 + 0.000000021 * t ** 3 + 0.000000007 * t ** 4
    s += pr
    h = 280.46645 + 36000.7697489 * t + 0.00030322222 * t ** 2 + 0.000000020 * t ** 3 - 0.00000000654 * t ** 4
    p = 83.35324312 + 4069.01363525 * t - 0.01032172222 * t ** 2 - 0.0000124991 * t ** 3 + 0.00000005263 * t ** 4
    zns = 234.95544499 + 1934.13626197 * t - 0.00207561111 * t ** 2 - 0.00000213944 * t ** 3 + 0.00000001650 * t ** 4
    ps = 282.93734098 + 1.71945766667 * t + 0.00045688889 * t ** 2 - 0.00000001778 * t ** 3 - 0.00000000334 * t ** 4

    s = s % 360.0
    h = h % 360.0
    p = p % 360.0
    zns = zns % 360.0
    ps = ps % 360.0

    rsta = np.sqrt(np.sum(xsta ** 2, axis=1))
    sinphi = xsta[:, 2] / rsta
    cosphi = np.sqrt(xsta[:, 0] ** 2 + xsta[:, 1] ** 2) / rsta
    cosla = xsta[:, 0] / cosphi / rsta
    sinla = xsta[:, 1] / cosphi / rsta

    xcorsta = np.zeros_like(xsta)

    dat = _DATDI_LON
    thetaf = (dat[:, 0] * s + dat[:, 1] * h + dat[:, 2] * p + dat[:, 3] * zns + dat[:, 4] * ps) * _DEG2RAD  # (5,)

    sin_tf = np.sin(thetaf)  # (5,) — no station dependence
    cos_tf = np.cos(thetaf)  # (5,)

    dR_ip = dat[:, 5]
    dT_ip = dat[:, 6]
    dR_op = dat[:, 7]
    dT_op = dat[:, 8]

    # dr and dn depend on station latitude
    p2_factor = 0.5 * (3.0 * sinphi ** 2 - 1.0)  # (N,)
    cs2_factor = cosphi * sinphi * 2.0  # (N,)

    # (5, N) via broadcasting — note: step2lon uses cos for ip, sin for op
    dr = (dR_ip[:, np.newaxis] * p2_factor * cos_tf[:, np.newaxis]
          + dR_op[:, np.newaxis] * p2_factor * sin_tf[:, np.newaxis])
    dn = (dT_ip[:, np.newaxis] * cs2_factor * cos_tf[:, np.newaxis]
          + dT_op[:, np.newaxis] * cs2_factor * sin_tf[:, np.newaxis])

    dr_tot = dr.sum(axis=0)
    dn_tot = dn.sum(axis=0)

    xcorsta[:, 0] = dr_tot * cosla * cosphi - dn_tot * sinphi * cosla
    xcorsta[:, 1] = dr_tot * sinla * cosphi - dn_tot * sinphi * sinla
    xcorsta[:, 2] = dr_tot * sinphi + dn_tot * cosphi

    return xcorsta / 1000.0  # mm to meters


# ---------------------------------------------------------------------------
# Integrated pytest tests — run with: pytest utils_tidal.py -v
# ---------------------------------------------------------------------------

def _pysolid_point(lon, lat, dt):
    """Reference: pySolid Fortran solid_grid at exact time."""
    from pysolid.solid import solid_grid as _sg
    te, tn, tu = _sg(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second,
                      lat, 1.0, 1, lon, 1.0, 1)
    return te[0, 0], tn[0, 0], tu[0, 0]


def _random_datetimes(n, seed=42):
    rng = np.random.default_rng(seed)
    start = datetime.datetime(2000, 1, 1)
    total_seconds = (datetime.datetime(2040, 1, 1) - start).total_seconds()
    offsets = rng.uniform(0, total_seconds, size=n)
    return [start + datetime.timedelta(seconds=float(s)) for s in offsets]


import datetime  # noqa: E402
import pytest  # noqa: E402


class TestVsPySolid:
    """Compare against pySolid (IERS 2003 Fortran reference)."""

    @pytest.fixture(scope="class")
    def global_grid(self):
        lons_1d = np.arange(-180, 180, 10.0)
        lats_1d = np.arange(-85, 86, 10.0)
        lon_grid, lat_grid = np.meshgrid(lons_1d, lats_1d)
        return lon_grid.ravel(), lat_grid.ravel()

    @pytest.mark.parametrize("lon,lat,dt", [
        (-115.5, 32.8,  datetime.datetime(2023, 2, 6, 0, 0, 0)),
        (-115.5, 32.8,  datetime.datetime(2023, 2, 6, 12, 30, 0)),
        (28.5,   37.0,  datetime.datetime(2023, 2, 6, 6, 0, 0)),
        (139.7,  35.7,  datetime.datetime(2023, 6, 21, 12, 0, 0)),
        (-70.0,  -33.4, datetime.datetime(2023, 9, 15, 3, 0, 0)),
        (0.0,    0.1,   datetime.datetime(2023, 1, 1, 0, 0, 0)),
        (180.0,  -85.0, datetime.datetime(2020, 3, 20, 18, 0, 0)),
        (-60.0,  60.0,  datetime.datetime(2035, 12, 31, 23, 59, 0)),
    ])
    def test_agrees_within_0_001mm(self, lon, lat, dt):
        e_py, n_py, u_py = solid_tide(np.array([lon]), np.array([lat]), dt)
        e_ps, n_ps, u_ps = _pysolid_point(lon, lat, dt)
        dmax = max(abs(e_py[0] - e_ps), abs(n_py[0] - n_ps), abs(u_py[0] - u_ps))
        assert dmax < 1e-6, (
            f"Diff {dmax:.2e} m at lon={lon} lat={lat} {dt}\n"
            f"  Python: E={e_py[0]:.9f} N={n_py[0]:.9f} U={u_py[0]:.9f}\n"
            f"  pySolid: E={e_ps:.9f} N={n_ps:.9f} U={u_ps:.9f}")

    def test_global_grid_random_times(self, global_grid):
        lons, lats = global_grid
        max_diff = 0.0
        for dt in _random_datetimes(10):
            dt = dt.replace(microsecond=0)
            e_py, n_py, u_py = solid_tide(lons, lats, dt)
            for idx in range(0, len(lons), 50):
                e_ps, n_ps, u_ps = _pysolid_point(lons[idx], lats[idx], dt)
                dmax = max(abs(e_py[idx] - e_ps), abs(n_py[idx] - n_ps), abs(u_py[idx] - u_ps))
                max_diff = max(max_diff, dmax)
        assert max_diff < 1e-6, f"Max diff {max_diff:.2e} m exceeds 0.001 mm"


class TestVectorized:
    def test_consistency(self):
        dt = datetime.datetime(2023, 2, 6, 12, 30, 0)
        lons = np.array([-115.5, 28.5, 96.5, 0.0, 139.7])
        lats = np.array([32.8, 37.0, 20.0, 0.1, 35.7])
        e_vec, n_vec, u_vec = solid_tide(lons, lats, dt)
        for j in range(len(lons)):
            e_pt, n_pt, u_pt = solid_tide(np.array([lons[j]]), np.array([lats[j]]), dt)
            assert abs(e_vec[j] - e_pt[0]) < 1e-12
            assert abs(n_vec[j] - n_pt[0]) < 1e-12
            assert abs(u_vec[j] - u_pt[0]) < 1e-12


class TestShape:
    def test_2d_preservation(self):
        dt = datetime.datetime(2023, 2, 6, 12, 30, 0)
        lons_2d, lats_2d = np.meshgrid(np.linspace(-116, -115, 5), np.linspace(32, 33, 4))
        e, n, u = solid_tide(lons_2d, lats_2d, dt)
        assert e.shape == (4, 5) and n.shape == (4, 5) and u.shape == (4, 5)


class TestMagnitude:
    @pytest.fixture(scope="class")
    def global_grid(self):
        lons_1d = np.arange(-180, 180, 10.0)
        lats_1d = np.arange(-85, 86, 10.0)
        lon_grid, lat_grid = np.meshgrid(lons_1d, lats_1d)
        return lon_grid.ravel(), lat_grid.ravel()

    def test_max_displacement_under_1m(self, global_grid):
        lons, lats = global_grid
        dt = datetime.datetime(2023, 6, 21, 12, 0, 0)
        e, n, u = solid_tide(lons, lats, dt)
        total = np.sqrt(e**2 + n**2 + u**2)
        assert total.max() < 1.0
        assert total.max() > 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
