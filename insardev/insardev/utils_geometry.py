# ----------------------------------------------------------------------------
# insardev
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2026, Alexey Pechnikov
#
# See the LICENSE file in the insardev directory for license terms.
# Professional use requires an active per-seat subscription at: https://patreon.com/pechnikov
# ----------------------------------------------------------------------------
"""THE RADAR GEOMETRY OF A GEOCODED BURST FROM PER-BURST NUMBERS: elevation() and elevation_phase() read the phase,
its x and y, and per-burst numbers only -- never azi, rng or ele; incidence() reads the transform's ele as each pixel's own height.

PER-BURST NUMBERS (one date's: the stack's REFERENCE date): near_range, rng_samp_rate, earth_radius,
SC_height_start/end, num_lines, num_rng_bins, radar_wavelength, PRF, lookdir, ref_height, the exact radar-extent polygon
`geometry` (of the crop, N116), the CRS, and the `ele` actual_range ATTRIBUTE: its midpoint h_attr, and its ends, the heights of
incidence()'s nodes.

MAPPING x, y -> radar, a satellite track rebuilt on WGS84 from those numbers and the polygon only:
  corners T_k (k = 0 near-early, 1 far-early, 2 far-late, 3 near-late) are the polygon's vertices, computed on the
  earth_radius sphere: T = WGS84(lat, lon, h_T) with |T| = earth_radius;
  zero-Doppler planes n . x = c at lines 0 and L - 1 through that line's near and far corners: the satellite position Rs s (s
  on the plane through the Earth centre and the corners, normal t) is moved along t by the length k (T_near . s - Rs),
  k = v_r / v, and normalised to the new s; n . s = v_r / |V| (|V| = sqrt(v^2 + v_r^2), v_r = (SC_height_end -
  SC_height_start) PRF / (L - 1), v = sqrt(GM / Rs)), n in span(t, e2) with e2 normal to T_far - T_near and t, c = n . T_near;
  the sub-satellite direction s on each plane at the Earth angle beta from the near corner,
      cos beta = (Rs^2 + |T_near|^2 - rho_n^2) / (2 Rs |T_near|),  Rs = earth_radius + SC_height at that line;
  a ground point G = WGS84(lat, lon, h): a_i = G.n_i - c_i (i = 0, 3), w = a0 / (a0 - a3), line = w (L - 1),
      s = slerp(s0, s3, w),
      rho = |G - (earth_radius + SC_height(line)) s|.
  (lat, lon) from x, y by pyproj at the absolute lattice nodes (multiples of STEP_M in map units, _lattice_step); everything per pair is bilinear between the nodes and
  applied in ONE numba pass.

KERNEL (k4 = 4 pi / lambda, f = line / (L - 1), Rs(f) = earth_radius + hs + (he - hs) f):
  ground radius R(h) = |P(lat, hg)| + kN (h - hg), kN from the WGS84 normal tilt eps and the look azimuth (the
  polygon's near-edge heading minus the look side's 90 deg);
  cos g = (rho^2 + Rs^2 - R^2) / (2 rho Rs), sin th = Rs sin g / R, incidence = acos(cos eps cos th + ...);
  each date's baseline (b_h, b_v) = B (cos alpha, sin alpha) at f = 0 and 1, linear in f;
  B_perp = b_h cos g + b_v sin g; m_k(h) the range change to date k, D = m_j - m_i.
  ONE DIVISION (tangent at h_s): out = A0 - phi / (k4 kN B_perp_pair / (rho sin th)), needs the pair DIFFERENCE.
  LINEARIZED JOINT SOLVE (needs both dates' own triples): the exact secant S(h) = -k4 (D(h) - D(anchor)) /
  (h - anchor) with the position moved to h, linear about h_a: S0 = S(h_a), S1 = dS/dh by +-JOINT_DELTA; then
  H = anchor - phi / S0 and JOINT_NITER times H = anchor - phi / (S0 + S1 (H - h_a)).
"""
import numpy as np
import numba as _numba
import threading
from .utils_circular import wrap_phase

C_LIGHT = 299792458.0
A_WGS = 6378137.0
F_WGS = 1.0 / 298.257223563
E2_WGS = F_WGS * (2.0 - F_WGS)
# the WGS84 gravitational parameter: the orbital speed sqrt(GM / Rs) of the satellite track
GM_WGS = 3.986004418e14
# node spacing in metres (degrees of a geographic CRS: _lattice_step): the nodes are the absolute multiples of it from
# zero, so a pixel's nodes never depend on the grid it is in; the geometry and the per-pair divisor at nodes, bilinear
# in between (STEP_M chosen by measurement, GMH U4)
STEP_M = 225.0
METRES_PER_DEGREE = 111320.0
# the linearized joint solve: its slope by a central difference of +-250 m, three fixed-point steps
JOINT_DELTA = 250.0
JOINT_NITER = 3
# the per-burst numbers the geometry reads (one date's)
SCALARS = ('near_range', 'rng_samp_rate', 'earth_radius', 'SC_height_start', 'SC_height_end', 'num_lines',
           'num_rng_bins', 'radar_wavelength', 'lookdir', 'geometry', 'PRF')
TRIPLES = ('baseline_start', 'baseline_center', 'baseline_end', 'alpha_start', 'alpha_center', 'alpha_end')


# ------------------------------------------------------------------------------------------------ points
def _geo2ecef(lat, lon, h):
    s, c = np.sin(lat), np.cos(lat)
    N = A_WGS / np.sqrt(1.0 - E2_WGS * s * s)
    return np.stack([(N + h) * c * np.cos(lon), (N + h) * c * np.sin(lon), (N * (1.0 - E2_WGS) + h) * s], axis=-1)


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


_TRANSFORMERS = {}


def _latlon(crs, X, Y):
    """Geodetic (lat, lon) in radians of map points."""
    from pyproj import Transformer
    tr = _TRANSFORMERS.get(crs)
    if tr is None:
        tr = _TRANSFORMERS[crs] = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    lon, lat = tr.transform(np.asarray(X, np.float64), np.asarray(Y, np.float64))
    return np.radians(np.asarray(lat, np.float64)), np.radians(np.asarray(lon, np.float64))


# ------------------------------------------------------------------------------------------------ per burst
def burst_geometry(values: dict, crs: str, h_ref: float, dem_mode: bool, h_attr: float) -> dict:
    """The per-burst model: one date's numbers (`values`, SCALARS), the CRS (WKT), the flat-earth reference height
    and mode, and the assumed terrain height h_attr (the ele actual_range midpoint, or 0). Plain numbers and small
    arrays: it travels to the dask workers with the graph. The satellite track is built here, once."""
    from shapely import wkt
    g = lambda n: float(values[n])
    P = dict(near=g('near_range'), dr=C_LIGHT / (2.0 * g('rng_samp_rate')), Re=g('earth_radius'),
             hs=g('SC_height_start'), he=g('SC_height_end'), L=g('num_lines'), NR=g('num_rng_bins'), PRF=g('PRF'),
             lam=g('radar_wavelength'))
    P['k4'] = 4.0 * np.pi / P['lam']
    P['side'] = 1.0 if str(values['lookdir']).upper().startswith('R') else -1.0
    P['dem_mode'] = bool(dem_mode)
    P['h_ref'] = 0.0 if dem_mode else float(h_ref)
    P['h_attr'] = float(h_attr)
    poly = wkt.loads(str(values['geometry']))
    lon, lat = (np.asarray(q, np.float64)[:4] for q in poly.exterior.coords.xy)
    latm = np.radians(0.5 * (lat[0] + lat[3]))
    psi = np.arctan2((lon[3] - lon[0]) * np.cos(latm), lat[3] - lat[0])
    P['alpha_s'] = float(psi - P['side'] * 0.5 * np.pi)
    P['crs'] = crs
    P.update(_track(P, np.radians(lat), np.radians(lon)))
    return P


def _track(P, lat, lon):
    """The satellite track from the polygon corners (radians) and the per-burst numbers. For each end line (0, L - 1), t is the
    unit normal of the plane through the Earth centre and the near and far corners (signed along s3 - s0). The sub-satellite
    direction s (angle beta from the near corner on that plane) gives the position Rs s, which is moved along t by the length
    k (T_near . s - Rs), k = v_r / v, and normalised: s = unit(Rs s + k (T_near . s - Rs) t). The zero-Doppler plane n . x = c
    through the near and far corners has n = cos(ph) t + sin(ph) e2 (e2 normal to T_far - T_near and t) with
    n . s = k / sqrt(1 + k^2) = v_r / |V|, and c = n . T_near. The satellite is not on its plane: Rs s is 43.2 m (Gastein)
    and 46.4 m (Turkiye) off it (measured). Om is the angle between s0 and s3."""
    Re = P['Re']
    h = np.zeros(4)
    for _ in range(4):
        h = h + (Re - np.linalg.norm(_geo2ecef(lat, lon, h), axis=-1))
    T = _geo2ecef(lat, lon, h)
    # the corner pixel centres: bins 0 and NR - 1 (near_range is the range of sample 0)
    rn = P['near']
    L = P['L']
    vr = (P['he'] - P['hs']) * P['PRF'] / (L - 1.0)
    out = {}
    Rs = {}
    for tag, iN, iF, a in (('0', 0, 1, 0.0), ('3', 3, 2, L - 1.0)):
        Rs[tag] = Re + P['hs'] + (P['he'] - P['hs']) * a / (L - 1.0)
        tN, tF = _unit(T[iN]), _unit(T[iF])
        RN = np.linalg.norm(T[iN])
        n = _unit(np.cross(tN, tF))
        e = _unit(tF - np.dot(tF, tN) * tN)
        bn = np.arccos((Rs[tag] * Rs[tag] + RN * RN - rn * rn) / (2 * Rs[tag] * RN))
        out['n' + tag] = n
        out['s' + tag] = np.cos(bn) * tN - np.sin(bn) * e
    d = out['s3'] - out['s0']
    for tag, iN, iF in (('0', 0, 1), ('3', 3, 2)):
        t = out['n' + tag] * np.sign(np.dot(out['n' + tag], d))
        k = vr / np.sqrt(GM_WGS / Rs[tag])
        s = _unit(Rs[tag] * out['s' + tag] + k * (np.dot(T[iN], out['s' + tag]) - Rs[tag]) * t)
        out['s' + tag] = s
        # the normal in the plane through the corners' line (normal to T_far - T_near), with N . s = v_r / |V|
        e2 = _unit(np.cross(T[iF] - T[iN], t))
        kp = k / np.sqrt(1.0 + k * k)
        A_, B_ = np.dot(t, s), np.dot(e2, s)
        R_ = np.hypot(A_, B_)
        ph = [np.arctan2(B_, A_) - np.arccos(kp / R_), np.arctan2(B_, A_) + np.arccos(kp / R_)]
        ph = min(ph, key=lambda p: abs(wrap_phase(p)))
        out['n' + tag] = np.cos(ph) * t + np.sin(ph) * e2
        out['c' + tag] = np.dot(out['n' + tag], T[iN])
    out['Om'] = float(np.arccos(np.dot(out['s0'], out['s3'])))
    return out


def _radar(P, G):
    """(line, rho, sat) of ECEF ground points G (..., 3) on the track: the line, the range and the satellite position (..., 3)."""
    a0 = G @ P['n0'] - P['c0']
    a3 = G @ P['n3'] - P['c3']
    w = a0 / (a0 - a3)
    L = P['L']
    line = w * (L - 1.0)
    Om = P['Om']
    s = (np.sin((1.0 - w) * Om)[..., None] * P['s0'] + np.sin(w * Om)[..., None] * P['s3']) / np.sin(Om)
    Rs = P['Re'] + P['hs'] + (P['he'] - P['hs']) * line / (L - 1.0)
    sat = Rs[..., None] * s
    rho = np.linalg.norm(G - sat, axis=-1)
    return line, rho, sat


def look_azimuth(P, lat, lon, hg):
    """The look azimuth (degrees, clockwise from geodetic north, in [0, 360)) of the line of sight from the satellite of the burst P
    to the points (lat, lon in radians) at the heights hg (m, WGS84): the satellite is taken on the zero-Doppler line of each point."""
    G = _geo2ecef(lat, lon, np.broadcast_to(np.asarray(hg, np.float64), np.shape(lat)))
    _, _, S = _radar(P, G)
    d = G - S
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    de = -so * d[..., 0] + co * d[..., 1]
    dn = -sl * co * d[..., 0] - sl * so * d[..., 1] + cl * d[..., 2]
    return np.mod(np.degrees(np.arctan2(de, dn)), 360.0)


def date_triples(baseline: dict) -> tuple:
    """(b_h, b_v), each (n, 2) at the scene start and end, from baseline_* (m) and alpha_* (deg), (n,) each: the
    baseline vectors B (cos alpha, sin alpha). The centre values are not read: the baseline is linear in between."""
    B = np.stack([np.asarray(baseline[f'baseline_{n}'], np.float64) for n in ('start', 'end')], axis=-1)
    a = np.radians(np.stack([np.asarray(baseline[f'alpha_{n}'], np.float64) for n in ('start', 'end')], axis=-1))
    return B * np.cos(a), B * np.sin(a)


# ------------------------------------------------------------------------------------------------ kernel
def _at(w, nodes):
    return w[0] * nodes[0] + w[1] * nodes[1]


class _Pix:
    """The geometry at nodes, the position (line, rho) at the height hg (WGS84)."""

    def __init__(self, P, lat, lon, hg):
        G = _geo2ecef(lat, lon, np.broadcast_to(np.asarray(hg, np.float64), np.shape(lat)))
        line, self.rho, _ = _radar(P, G)
        self.P = P
        self.f = line / (P['L'] - 1.0)
        self.hg = np.broadcast_to(np.asarray(hg, np.float64), self.rho.shape)
        self.Rs = P['Re'] + P['hs'] + (P['he'] - P['hs']) * self.f
        sl, cl = np.sin(lat), np.cos(lat)
        N = A_WGS / np.sqrt(1.0 - E2_WGS * sl * sl)
        pxy = (N + self.hg) * cl
        pz = (N * (1.0 - E2_WGS) + self.hg) * sl
        self.Rg = np.hypot(pxy, pz) - self.hg
        eps = lat - np.arctan2(pz, pxy)
        ce, se = np.cos(eps), np.sin(eps)
        _, _, sth, cth = self._tri(self.Rg + self.hg)
        ca = np.cos(P['alpha_s'])
        self.kN = sth / (ce * sth - se * cth * ca)
        self.inc = np.arccos(ce * cth + se * sth * ca)
        self.w = (1.0 - self.f, self.f)

    def _tri(self, R):
        rho, Rs = self.rho, self.Rs
        cg = (rho * rho + Rs * Rs - R * R) / (2 * rho * Rs)
        sg = np.sqrt(1 - cg * cg)
        sth = Rs * sg / R
        cth = np.sqrt(1 - sth * sth)
        return cg, sg, sth, cth

    def R(self, h):
        return self.Rg + self.hg + self.kN * (h - self.hg)

    def m(self, bh, bv, h):
        """The range change to a date of baseline (bh, bv) at the height h on this position's range circle, and
        its derivative by h."""
        R = self.R(h)
        rho, Rs = self.rho, self.Rs
        c = (rho * rho + Rs * Rs - R * R) / (2 * rho * Rs)
        s = np.sqrt(1 - c * c)
        bpar = s * bh - c * bv
        bperp = c * bh + s * bv
        B2 = bh * bh + bv * bv
        rk = np.sqrt(rho * rho + B2 - 2 * rho * bpar)
        return (B2 - 2 * rho * bpar) / (rho + rk), -bperp * R * self.kN / (Rs * s * rk)

    def tan_fac(self, h):
        """k4 kN / (rho sin th) and (cos g, sin g) at the height h: radians per metre of height per metre of
        perpendicular baseline."""
        cg, sg, sth, _ = self._tri(self.R(h))
        return self.P['k4'] * self.kN / (self.rho * sth), cg, sg


def _tangent_K(px, h, bi, bj):
    """ONE DIVISION, out = A0 - phi / K: K = k4 kN B_perp_pair / (rho sin th) at the height h; the DIFFERENCE of
    the two dates' triples only."""
    fac, cg, sg = px.tan_fac(h)
    bh = _at(px.w, bj[0]) - _at(px.w, bi[0])
    bv = _at(px.w, bj[1]) - _at(px.w, bi[1])
    return fac * (bh * cg + bv * sg)


def _secant(px, h, anchor, bi, bj):
    """The exact secant divisor between the anchor and h on the range circle of px, K = -k4 (D(h) - D(anchor)) /
    (h - anchor) with D = m_j - m_i (so H = anchor - phi / K is exact for H = h); the tangent where
    |h - anchor| < 0.5 m. Needs both dates' own triples."""
    P = px.P
    bhi, bvi = _at(px.w, bi[0]), _at(px.w, bi[1])
    bhj, bvj = _at(px.w, bj[0]), _at(px.w, bj[1])
    vj, gj = px.m(bhj, bvj, h)
    vi, gi = px.m(bhi, bvi, h)
    rj, _ = px.m(bhj, bvj, anchor)
    ri, _ = px.m(bhi, bvi, anchor)
    dh = h - anchor
    near = np.abs(dh) < 0.5
    return np.where(near, -P['k4'] * (gj - gi), -P['k4'] * ((vj - vi) - (rj - ri)) / np.where(near, 1.0, dh))


# ------------------------------------------------------------------------------------------------ nodes
def _lattice_step(crs):
    """The node spacing of a CRS in its own units: STEP_M metres, or the same length in degrees for a geographic CRS.
    Absolute: the nodes are its multiples from zero, so no grid, crop or pixel spacing enters."""
    from pyproj import CRS
    return STEP_M / METRES_PER_DEGREE if CRS.from_user_input(crs).is_geographic else STEP_M


def _lattice(v, S):
    """The absolute lattice of spacing S (map units): per map coordinate v the node j = floor(v / S) (at the position j S)
    and the weight w = v / S - j of the node j + 1. A pixel's weights depend on its own coordinate only."""
    t = np.asarray(v, np.float64) / S
    j = np.floor(t)
    return j.astype(np.int64), t - j


def _node_latlon(P, gy, gx, ycoord, xcoord):
    """The geometry nodes of a block of global pixel indices gy, gx of the grid whose axes are ycoord, xcoord (map units):
    the multiples of _lattice_step from zero (_lattice), the block's own nodes and the next one of each axis. Returns (jy,
    wy, jx, wx, lat, lon): per block row and column the node index from the block's first node and the weight of the next
    node; the node latitudes and longitudes (radians, shape (nodes y, nodes x)). The nodes of a pixel do not depend on the
    grid or the crop it is in."""
    S = _lattice_step(P['crs'])
    spans = []
    for axis, g in ((ycoord, gy), (xcoord, gx)):
        j, w = _lattice(np.asarray(axis, np.float64)[g], S)
        k0 = int(j.min())
        spans.append((j - k0, w, k0, int(j.max()) - k0 + 2))
    (jy, wy, ky, ny), (jx, wx, kx, nx) = spans
    Y, X = np.meshgrid((ky + np.arange(ny)) * S, (kx + np.arange(nx)) * S, indexing='ij')
    lat, lon = _latlon(P['crs'], X, Y)
    return jy, wy, jx, wx, lat, lon


def _upsample_x(Kc, jx, wx):
    """(..., ny_nodes, nx_nodes) -> (..., ny_nodes, nx), linear in x at the lattice positions jx (left node), wx."""
    return Kc[..., :, jx] * (1 - wx) + Kc[..., :, jx + 1] * wx


def _bilinear(Kc, jy, wy, jx, wx):
    """Bilinear upsampling of Kc (ny_nodes, nx_nodes) at the lattice positions (jy, wy), (jx, wx) to (ny, nx), float32."""
    Kx = _upsample_x(Kc, jx, wx).astype(np.float32)
    w = np.asarray(wy, np.float32)[:, None]
    return Kx[jy, :] * (1 - w) + Kx[jy + 1, :] * w


@_numba.njit(nogil=True, cache=True)
def _apply_division(ph, Kx, jy, wy, A0, out):
    """out[q, r, c] = A0[q] - ph[q, r, c] / K, K bilinear from the x-upsampled node rows: ONE pass. Serial, as
    every numba kernel of the library: dask parallelises across blocks."""
    npair, ny, nx = ph.shape
    for q in range(npair):
        a0 = A0[q]
        for r in range(ny):
            j = jy[r]
            b = wy[r]
            a = 1.0 - b
            for c in range(nx):
                K = Kx[q, j, c] * a + Kx[q, j + 1, c] * b
                out[q, r, c] = a0 - ph[q, r, c] / K
    return out


@_numba.njit(nogil=True, cache=True)
def _apply_joint(ph, S0x, S1x, jy, wy, anchor, ha, off, niter, out):
    """The linearized joint solve: H = anchor - phi / (S0 + S1 (H - ha)), niter fixed-point steps from
    H = anchor - phi / S0; S0, S1 bilinear from the x-upsampled node rows; out = H - off. ONE pass."""
    npair, ny, nx = ph.shape
    for q in range(npair):
        for r in range(ny):
            j = jy[r]
            b = wy[r]
            a = 1.0 - b
            for c in range(nx):
                s0 = S0x[q, j, c] * a + S0x[q, j + 1, c] * b
                s1 = S1x[q, j, c] * a + S1x[q, j + 1, c] * b
                p = ph[q, r, c]
                H = anchor - p / s0
                for _ in range(niter):
                    H = anchor - p / (s0 + s1 * (H - ha))
                out[q, r, c] = H - off
    return out


def _warmup_numba_cache():
    """Compile the numba kernels once in the main process so dask workers load them from the cache."""
    ph = np.zeros((1, 2, 2), np.float32)
    jy = np.zeros(2, np.int64)
    _apply_division(ph, np.ones((1, 2, 2), np.float32), jy, np.zeros(2, np.float32), np.zeros(1, np.float32),
                    np.empty((1, 2, 2), np.float32))
    _apply_joint(ph, np.ones((1, 2, 2)), np.zeros((1, 2, 2)), jy, np.zeros(2), 0.0, 0.0, 0.0, 3,
                 np.empty((1, 2, 2), np.float32))


_warmup_numba_cache()


# ------------------------------------------------------------------------------------------------ blocks
def elevation_block(ph, pidx, gy, gx, *, geos, pairs, joint, ycoord, xcoord):
    """One phase block (npair, ny, nx) -> elevation, from the phase and the block's GLOBAL pixel indices gy, gx (its
    place in the whole grid, whose axes are ycoord, xcoord): the lattice is the absolute one (_node_latlon), so the result
    does not depend on the block or on the crop.

    geos: the per-burst models (burst_geometry()); pairs: per pair of the variable (geometry index, b_i, b_j),
    b = (b_h nodes, b_v nodes), the dates' own triples (joint) or 0 and the pair difference (one division).
    joint: True the linearized joint solve (DEM mode: anchor and h_a the ele actual_range midpoint, the output the
    residual; flat mode: anchor the reference height, h_a the midpoint), False one division (DEM mode: position and
    tangent at h_attr; flat mode: position at h_attr, tangent at (h_attr + h_ref) / 2)."""
    sel = [pairs[int(p)] for p in np.asarray(pidx).ravel()]
    P0 = geos[sel[0][0]]
    jy, wy, jx, wx, lat, lon = _node_latlon(P0, gy, gx, ycoord, xcoord)
    cache = {}

    def pix(g, h):
        k = (g, float(h))
        if k not in cache:
            cache[k] = _Pix(geos[g], lat, lon, h)
        return cache[k]
    phc = np.ascontiguousarray(ph, np.float32)
    out = np.empty(ph.shape, np.float32)
    if joint:
        S0, S1 = [], []
        for g, bi, bj in sel:
            P = geos[g]
            ha = P['h_attr']
            anchor = ha if P['dem_mode'] else P['h_ref']
            S = [_secant(pix(g, h), h, anchor, bi, bj) for h in (ha, ha + JOINT_DELTA, ha - JOINT_DELTA)]
            S0.append(S[0])
            S1.append((S[1] - S[2]) / (2.0 * JOINT_DELTA))
        S0x = _upsample_x(np.stack(S0), jx, wx)
        S1x = _upsample_x(np.stack(S1), jx, wx)
        # one burst model per variable on the joint path (the stack's): its anchor and offset
        P = geos[sel[0][0]]
        ha = P['h_attr']
        anchor, off = (ha, ha) if P['dem_mode'] else (P['h_ref'], 0.0)
        return _apply_joint(phc, np.ascontiguousarray(S0x, np.float64), np.ascontiguousarray(S1x, np.float64),
                            jy, np.ascontiguousarray(wy, np.float64), float(anchor), float(ha), float(off),
                            int(JOINT_NITER), out)
    K, A0 = [], []
    for g, bi, bj in sel:
        P = geos[g]
        ha = P['h_attr']
        hs = ha if P['dem_mode'] else 0.5 * (ha + P['h_ref'])
        K.append(_tangent_K(pix(g, ha), hs, bi, bj))
        A0.append(0.0 if P['dem_mode'] else P['h_ref'])
    Kx = _upsample_x(np.stack(K), jx, wx)
    return _apply_division(phc, np.ascontiguousarray(Kx, np.float32), jy, np.ascontiguousarray(wy, np.float32),
                           np.asarray(A0, np.float32), out)


def _own_height(nodes, ele, heights):
    """The block at each pixel's own height ele (y, x) from its nodes: (a, m, b) at lo, (lo + hi) / 2 and hi, heights =
    (lo, hi), the parabola through them in float64; (a,) when lo == hi; a NaN ele gives NaN.

    >>> _own_height((a, m, b), ele, (lo, hi))"""
    lo, hi = (float(h) for h in heights)
    e = np.asarray(ele, np.float64)
    if hi == lo:
        return np.where(np.isfinite(e), nodes[0], np.nan)
    a, m, b = (np.asarray(n, np.float64) for n in nodes)
    t = (e - lo) / (hi - lo)
    # the parabola through (0, a), (1/2, m), (1, b) at t: a + t ((b - a) + 2 (t - 1) (a + b - 2 m))
    q = (a + b) - 2.0 * m
    q *= 2.0 * (t - 1.0)
    q += b - a
    q *= t
    q += a
    return q


def incidence_block(gy, gx, ele=None, *, geo, ycoord, xcoord, heights=None):
    """The incidence (radians, float32) of a (y, x) block from its global pixel indices gy, gx and the whole axes ycoord, xcoord
    (map units; the lattice is absolute, see _node_latlon). Without ele: at the terrain height h_attr. With the block's ele and
    heights = (lo, hi), the ends of the ele actual_range: at each pixel's own height (_own_height, nodes at lo, the midpoint, hi).

    >>> incidence_block(gy, gx, geo=geo, ycoord=y, xcoord=x)
    >>> incidence_block(gy, gx, ele, geo=geo, ycoord=y, xcoord=x, heights=(lo, hi))"""
    jy, wy, jx, wx, lat, lon = _node_latlon(geo, gy, gx, ycoord, xcoord)
    if ele is None:
        return _bilinear(_Pix(geo, lat, lon, geo['h_attr']).inc, jy, wy, jx, wx)
    lo, hi = (float(h) for h in heights)
    hs = (lo,) if hi == lo else (lo, 0.5 * (lo + hi), hi)
    nodes = [_bilinear(_Pix(geo, lat, lon, h).inc, jy, wy, jx, wx) for h in hs]
    return _own_height(nodes, ele, (lo, hi)).astype(np.float32)


def look_block(gy, gx, ele=None, *, geo, ycoord, xcoord, heights=None):
    """The look azimuth (degrees, in [0, 360), float32) of a (y, x) block from its global pixel indices gy, gx and the whole axes
    ycoord, xcoord: the azimuth on the nodes, bilinear in its sine and cosine (no wrap at 0 and 360). Without ele: at the terrain
    height h_attr. With the block's ele and heights = (lo, hi), the ends of the ele actual_range: at each pixel's own height
    (_own_height of the sine and cosine, as incidence_block).

    >>> look_block(gy, gx, geo=geo, ycoord=y, xcoord=x)
    >>> look_block(gy, gx, ele, geo=geo, ycoord=y, xcoord=x, heights=(lo, hi))"""
    jy, wy, jx, wx, lat, lon = _node_latlon(geo, gy, gx, ycoord, xcoord)

    def sc(h):
        a = np.radians(look_azimuth(geo, lat, lon, h))
        return np.stack([_bilinear(np.sin(a), jy, wy, jx, wx), _bilinear(np.cos(a), jy, wy, jx, wx)])
    if ele is None:
        s, c = sc(geo['h_attr'])
    else:
        lo, hi = (float(h) for h in heights)
        hs = (lo,) if hi == lo else (lo, 0.5 * (lo + hi), hi)
        s, c = _own_height([sc(h) for h in hs], ele, (lo, hi))
    return np.mod(np.degrees(np.arctan2(s, c)), 360.0).astype(np.float32)


def elevation_phase_block(gy, gx, *, geo, ycoord, xcoord):
    """k4 kN / (rho sin th) of a (y, x) block at the terrain height h_attr: radians of phase per metre of height
    per metre of perpendicular baseline, the one-division sensitivity elevation() divides by. The block is given by
    its global pixel indices gy, gx and the whole axes ycoord, xcoord."""
    jy, wy, jx, wx, lat, lon = _node_latlon(geo, gy, gx, ycoord, xcoord)
    return _bilinear(_Pix(geo, lat, lon, geo['h_attr']).tan_fac(geo['h_attr'])[0], jy, wy, jx, wx)


def _ref_mode(values) -> tuple:
    """(reference height, DEM mode) from a burst's ref_height values, the rule of _ref_height_stored(): NaN on
    every date is DEM mode (height 0.0), else the finite value of largest magnitude is the flat-earth height."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    return (float(v[np.argmax(np.abs(v))]), False) if v.size else (0.0, True)


def _horn_gradient_2d_for_dask(block, dy=1.0, dx=1.0):
    """Horn (1981) 3 x 3 terrain gradient of a DEM block, for map_overlap.

    Module level so dask ships it by name. The block arrives WITH its halo and
    leaves the same shape, so dask trims the depth back off.

    The two derivatives are packed into ONE complex array -- gx + 1j*gy, per
    metre of easting and northing -- because they come from the same nine
    reads. Slope and aspect are then elementwise on the result and cost no
    second pass over the raster.

    `dy`/`dx` are SIGNED coordinate steps per index, so northing running either
    way down the array gives the same gradient.
    """
    import numpy as np
    z = np.asarray(block, dtype=np.float64)
    flat = z.ndim == 3
    if flat:
        z = z[0]
    out = np.full(z.shape, np.nan, dtype=np.complex64)
    if z.shape[0] >= 3 and z.shape[1] >= 3:
        a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
        d, f = z[1:-1, :-2], z[1:-1, 2:]
        g, h, i = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
        # each side sums to four weights over a two-step span, hence the 8
        gx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * dx)
        gy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8.0 * dy)
        out[1:-1, 1:-1] = (gx + 1j * gy).astype(np.complex64)
    return out[None] if flat else out


# ------------------------------------------------------------------------------------------------ batch-free geometry
# The algorithm of the geometry methods of Batch, BatchCore and Stack (group GKK, unit U5): the batch classes keep the
# per-burst dispatch, the coordinates, the attributes and the argument checks, and call these functions on arrays.

def centre_geometry(px, py, xc, yc, n_lines, n_rng, near_range, rng_rate, sc_height, ref_height, earth_radius):
    """The slant range (m) and sin(incidence) at the centre of the PROCESSED area of one burst, on the sphere of
    earth_radius at ref_height. px, py: the four corners of the burst's radar extent (line and bin (0, 0), (0, n_rng-1),
    (n_lines-1, n_rng-1), (n_lines-1, 0)) in the burst CRS; xc, yc: the centre of the burst's grid. The centre's range bin
    comes from the inverse of the corner mapping and is kept only when it falls inside the burst; else the middle bin."""
    c_light = 299792458.0
    rng_c = n_rng / 2.0
    a_c = np.array([0.0, 0.0, n_lines - 1, n_lines - 1])
    r_c = np.array([0.0, n_rng - 1, n_rng - 1, 0.0])
    M, b = [], []
    for x_, y_, a_, r_ in zip(px, py, a_c, r_c):
        M.append([x_, y_, 1, 0, 0, 0, -a_ * x_, -a_ * y_]); b.append(a_)
        M.append([0, 0, 0, x_, y_, 1, -r_ * x_, -r_ * y_]); b.append(r_)
    h = np.linalg.solve(np.asarray(M, float), np.asarray(b, float))
    den = h[6] * xc + h[7] * yc + 1.0
    cand = (h[3] * xc + h[4] * yc + h[5]) / den
    if np.isfinite(cand) and 0.0 <= cand <= n_rng:
        rng_c = float(cand)

    slant_range = near_range + rng_c * (c_light / (2.0 * rng_rate))
    ground_dist = earth_radius + ref_height
    sat_dist = earth_radius + sc_height
    cos_earth = np.clip(
        (ground_dist ** 2 + sat_dist ** 2 - slant_range ** 2)
        / (2.0 * ground_dist * sat_dist), -1.0, 1.0)
    sin_inc = np.clip(
        sat_dist * np.sin(np.arccos(cos_earth)) / slant_range, -1.0, 1.0)
    return slant_range, sin_inc


def elevation_phase_factor(wavelength, slant_range, sin_inc):
    """Elevation phase per metre of height per metre of baseline (radians): (4 pi / wavelength) / (slant range sin(incidence))."""
    return (4.0 * np.pi / wavelength) / (slant_range * sin_inc)


def incidence_from_sine(sin_inc):
    """The incidence angle (radians) from its sine: arcsin(sin_inc)."""
    return float(np.arcsin(sin_inc))


def phase_to_los(phase, wavelength):
    """Line-of-sight displacement (metres, float32) of unwrapped phase (radians). The constant is negative to make
    LOS = -1 * range change."""
    # scale factor from phase in radians to displacement in meters
    # constant is negative to make LOS = -1 * range change
    scale = -float(wavelength) / (4 * np.pi)
    return (phase * scale).astype('float32')


def los_component(los, incidence, func):
    """LOS displacement divided by a function of the incidence (func = cos for vertical, sin for east-west), float32.
    incidence: a scalar (the burst-centre value, one per burst) or an array on the data's own grid."""
    if np.ndim(incidence) == 0:
        return (los / float(func(incidence))).astype('float32')
    return (los / func(incidence)).astype('float32')


def elevation_approx(phase, factor, baseline):
    """Height (metres) of a scalar or array phase from the burst-centre factor (radians per metre of height per metre
    of baseline): -phase / (factor * baseline), rounded to 3 decimals. A scalar gives a float, an array an array. A scalar
    phase has no pixel and no reference surface: it is a phase DIFFERENCE and maps to a signed height DIFFERENCE."""
    phase_arr = np.asarray(phase)
    is_scalar = phase_arr.ndim == 0
    # phi = -fac * B_perp * dh  ->  dh = -phi / (fac * B_perp), the same relation as the grids
    elev = -phase_arr / (factor * baseline) + 0.0  # + 0.0 turns -0.0 into 0.0
    if is_scalar:
        return round(float(elev), 3)
    return np.round(elev, 3)


# rasterio rasterizes in memory inside warnings.catch_warnings, which is process-wide: dask threads entering it together leak its
# NotGeoreferencedWarning to the display (rasterio 1.4.3, GDAL 3.9.3); one rasterization at a time per process keeps the values
_GEOMETRY_MASK_LOCK = threading.Lock()


def geometry_mask(shape, x0, y0, dx, dy, geoms):
    """The polygon test of one block of a burst: True where the pixel centre lies inside the geometries, rasterized as
    rio.clip does (rasterio geometry_mask, all_touched=False). shape: (rows, columns) of the block; x0, y0: the coordinates
    of its first pixel centre; dx, dy: the resolution of the burst's x and y axes."""
    from rasterio import features
    from rasterio.transform import Affine
    transform = Affine(dx, 0.0, x0 - dx / 2, 0.0, dy, y0 - dy / 2)
    with _GEOMETRY_MASK_LOCK:
        return features.geometry_mask(geoms, out_shape=tuple(shape), transform=transform,
                                      all_touched=False, invert=True)


def reproject_bbox(bounds, src_crs, dst_crs, n_edge):
    """The bounding box (minx, miny, maxx, maxy) in dst_crs of a box given in src_crs: the four EDGES sampled at n_edge
    points each, not its four corners. A lon/lat rectangle is not a rectangle in the batch CRS, and its extreme x/y can
    lie inside a curved edge."""
    from pyproj import CRS, Transformer
    minx, miny, maxx, maxy = bounds
    t = np.linspace(0.0, 1.0, n_edge)
    xs = np.r_[minx + (maxx - minx) * t, np.full(t.size, maxx), maxx - (maxx - minx) * t, np.full(t.size, minx)]
    ys = np.r_[np.full(t.size, miny), miny + (maxy - miny) * t, np.full(t.size, maxy), maxy - (maxy - miny) * t]
    xs, ys = Transformer.from_crs(CRS.from_user_input(src_crs), CRS.from_user_input(dst_crs),
                                  always_xy=True).transform(xs, ys)
    return np.min(xs), np.min(ys), np.max(xs), np.max(ys)
