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
"""THE RADAR GEOMETRY OF A GEOCODED BURST WITHOUT ANY TRANSFORM RASTER: elevation(), elevation_phase() and
incidence() read the phase, its x and y, and per-burst numbers only -- never azi, rng or ele.

PER-BURST NUMBERS (one date's: the stack's REFERENCE date): near_range, rng_samp_rate, earth_radius,
SC_height_start/end, num_lines, num_rng_bins, radar_wavelength, lookdir, ref_height, the exact radar-extent polygon
`geometry` (of the crop, N116), the CRS, and the midpoint of the `ele` actual_range ATTRIBUTE (h_attr).

MAPPING x, y -> radar, a satellite track rebuilt on WGS84 from those numbers and the polygon only:
  corners T_k (k = 0 near-early, 1 far-early, 2 far-late, 3 near-late) are the polygon's vertices, computed on the
  earth_radius sphere: T = WGS84(lat, lon, h_T) with |T| = earth_radius;
  zero-Doppler planes at lines 0 and L - 1 through the Earth centre and that line's near and far corners;
  the sub-satellite direction s on each plane at the Earth angle beta from the near corner,
      cos beta = (Rs^2 + |T_near|^2 - rho_n^2) / (2 Rs |T_near|),  Rs = earth_radius + SC_height at that line;
  a ground point G = WGS84(lat, lon, h): w = G.n0 / (G.n0 - G.n3), line = w (L - 1), s = slerp(s0, s3, w),
      rho = |G - (earth_radius + SC_height(line)) s|.
  (lat, lon) from x, y by pyproj at nodes every STEP pixels; everything per pair is bilinear between the nodes and
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

C_LIGHT = 299792458.0
A_WGS = 6378137.0
F_WGS = 1.0 / 298.257223563
E2_WGS = F_WGS * (2.0 - F_WGS)
# node spacing (pixels): the geometry and the per-pair divisor at nodes, bilinear in between (node error measured
# <= 9.3e-6 relative at 128 px)
STEP = 128
# the linearized joint solve: its slope by a central difference of +-250 m, three fixed-point steps
JOINT_DELTA = 250.0
JOINT_NITER = 3
# the per-burst numbers the geometry reads (one date's)
SCALARS = ('near_range', 'rng_samp_rate', 'earth_radius', 'SC_height_start', 'SC_height_end', 'num_lines',
           'num_rng_bins', 'radar_wavelength', 'lookdir', 'geometry')
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
             hs=g('SC_height_start'), he=g('SC_height_end'), L=g('num_lines'), NR=g('num_rng_bins'),
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
    """The satellite track: the zero-Doppler plane normals n0, n3 at lines 0 and L - 1, the sub-satellite
    directions s0, s3 and their angle Om, from the polygon corners (radians) and the per-burst numbers."""
    Re = P['Re']
    h = np.zeros(4)
    for _ in range(4):
        h = h + (Re - np.linalg.norm(_geo2ecef(lat, lon, h), axis=-1))
    T = _geo2ecef(lat, lon, h)
    # the corner pixel centres: bins 0 and NR - 1 (near_range is the range of sample 0)
    rn = P['near']
    L = P['L']
    out = {}
    for tag, iN, iF, a in (('0', 0, 1, 0.0), ('3', 3, 2, L - 1.0)):
        Rs = Re + P['hs'] + (P['he'] - P['hs']) * a / (L - 1.0)
        tN, tF = _unit(T[iN]), _unit(T[iF])
        RN = np.linalg.norm(T[iN])
        n = _unit(np.cross(tN, tF))
        e = _unit(tF - np.dot(tF, tN) * tN)
        bn = np.arccos((Rs * Rs + RN * RN - rn * rn) / (2 * Rs * RN))
        out['n' + tag] = n
        out['s' + tag] = np.cos(bn) * tN - np.sin(bn) * e
    d = out['s3'] - out['s0']
    out['n0'] = out['n0'] * np.sign(np.dot(out['n0'], d))
    out['n3'] = out['n3'] * np.sign(np.dot(out['n3'], d))
    out['Om'] = float(np.arccos(np.dot(out['s0'], out['s3'])))
    return out


def _radar(P, G):
    """(line, rho) of ECEF ground points G (..., 3) on the track."""
    a0 = G @ P['n0']
    a3 = G @ P['n3']
    w = a0 / (a0 - a3)
    L = P['L']
    line = w * (L - 1.0)
    Om = P['Om']
    s = (np.sin((1.0 - w) * Om)[..., None] * P['s0'] + np.sin(w * Om)[..., None] * P['s3']) / np.sin(Om)
    Rs = P['Re'] + P['hs'] + (P['he'] - P['hs']) * line / (L - 1.0)
    rho = np.linalg.norm(G - Rs[..., None] * s, axis=-1)
    return line, rho


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
        line, self.rho = _radar(P, G)
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
def _nodes(n, step):
    i = np.arange(0, n, step)
    if i[-1] != n - 1:
        i = np.append(i, n - 1)
    return i


def _interp_w(nodes, n):
    """Index of the left node and the weight of the right one, per output position 0 .. n - 1."""
    q = np.arange(n)
    j = np.clip(np.searchsorted(nodes, q, side='right') - 1, 0, nodes.size - 2)
    w = (q - nodes[j]) / (nodes[j + 1] - nodes[j])
    return j, w


def _node_latlon(P, yc, xc):
    ny, nx = np.size(yc), np.size(xc)
    iy = _nodes(ny, STEP)
    ix = _nodes(nx, STEP)
    Yn, Xn = np.meshgrid(np.asarray(yc, np.float64)[iy], np.asarray(xc, np.float64)[ix], indexing='ij')
    lat, lon = _latlon(P['crs'], Xn, Yn)
    return iy, ix, lat, lon


def _upsample_x(Kc, ix, nx):
    """(..., ny_nodes, nx_nodes) -> (..., ny_nodes, nx), linear in x."""
    if ix.size == 1:
        return np.repeat(Kc[..., :, :1], nx, axis=-1)
    jx, wx = _interp_w(ix, nx)
    return Kc[..., :, jx] * (1 - wx) + Kc[..., :, jx + 1] * wx


def _rows(iy, ny, Kx):
    """The row interpolation the numba pass applies: (row index, weight) per output row, and the x-upsampled
    node rows with a single node row doubled."""
    if iy.size == 1:
        return np.zeros(ny, np.int64), np.zeros(ny), np.concatenate([Kx, Kx], axis=-2)
    jy, wy = _interp_w(iy, ny)
    return jy.astype(np.int64), wy, Kx


def _bilinear(Kc, iy, ix, ny, nx):
    """Bilinear upsampling of Kc (ny_nodes, nx_nodes) at node indices iy, ix to (ny, nx), float32."""
    Kx = _upsample_x(Kc, ix, nx).astype(np.float32)
    if iy.size == 1:
        return np.repeat(Kx[:1, :], ny, axis=0)
    jy, wy = _interp_w(iy, ny)
    wy = wy.astype(np.float32)[:, None]
    return Kx[jy, :] * (1 - wy) + Kx[jy + 1, :] * wy


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
def elevation_block(ph, pidx, yc, xc, *, geos, pairs, joint):
    """One phase block (npair, ny, nx) -> elevation, from the phase and the block's y, x only.

    geos: the per-burst models (burst_geometry()); pairs: per pair of the variable (geometry index, b_i, b_j),
    b = (b_h nodes, b_v nodes), the dates' own triples (joint) or 0 and the pair difference (one division).
    joint: True the linearized joint solve (DEM mode: anchor and h_a the ele actual_range midpoint, the output the
    residual; flat mode: anchor the reference height, h_a the midpoint), False one division (DEM mode: position and
    tangent at h_attr; flat mode: position at h_attr, tangent at (h_attr + h_ref) / 2)."""
    sel = [pairs[int(p)] for p in np.asarray(pidx).ravel()]
    ny, nx = ph.shape[-2:]
    P0 = geos[sel[0][0]]
    iy, ix, lat, lon = _node_latlon(P0, yc, xc)
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
        jy, wy, S0x = _rows(iy, ny, _upsample_x(np.stack(S0), ix, nx))
        _, _, S1x = _rows(iy, ny, _upsample_x(np.stack(S1), ix, nx))
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
    jy, wy, Kx = _rows(iy, ny, _upsample_x(np.stack(K), ix, nx))
    return _apply_division(phc, np.ascontiguousarray(Kx, np.float32), jy, np.ascontiguousarray(wy, np.float32),
                           np.asarray(A0, np.float32), out)


def incidence_block(yc, xc, *, geo):
    """The incidence (radians) of a (y, x) block at the terrain height h_attr, from its y, x only."""
    ny, nx = np.size(yc), np.size(xc)
    iy, ix, lat, lon = _node_latlon(geo, yc, xc)
    return _bilinear(_Pix(geo, lat, lon, geo['h_attr']).inc, iy, ix, ny, nx)


def elevation_phase_block(yc, xc, *, geo):
    """k4 kN / (rho sin th) of a (y, x) block at the terrain height h_attr: radians of phase per metre of height
    per metre of perpendicular baseline, the one-division sensitivity elevation() divides by."""
    ny, nx = np.size(yc), np.size(xc)
    iy, ix, lat, lon = _node_latlon(geo, yc, xc)
    return _bilinear(_Pix(geo, lat, lon, geo['h_attr']).tan_fac(geo['h_attr'])[0], iy, ix, ny, nx)
