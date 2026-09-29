"""
Boundary geometry: point-to-polyline distance and side-of-line tests.

Pure numpy, no per-call object construction.  The scalar functions keep the
signatures the rest of the code base has always used; the ``*_batch``
functions do the same maths for (N, 2) arrays of [lat, lon] fixes.

Coordinates are treated as planar (lat, lon) degrees, exactly as the earlier
shapely implementation did, and distances are converted with 111.32 km/deg.
That is a deliberate simplification: over the 2 x 2 degree study area at 9 N
the anisotropy is under 2 %, and it is applied identically to every policy.
"""

import numpy as np

KM_PER_DEG = 111.32

_SEG_CACHE = {}


def _segments(points):
    """Cached segment arrays for a polyline given as [[lat, lon], ...]."""
    key = tuple(tuple(float(v) for v in p) for p in points)
    seg = _SEG_CACHE.get(key)
    if seg is None:
        P = np.asarray(key, dtype=np.float64)           # (M, 2) [lat, lon]
        A, B = P[:-1], P[1:]
        D = B - A
        L2 = np.sum(D * D, axis=1)
        L2 = np.where(L2 == 0, 1.0, L2)
        seg = (A, B, D, L2)
        _SEG_CACHE[key] = seg
    return seg


def _nearest(latlon, points):
    """(dist_deg (N,), idx (N,)) of the nearest segment for each fix."""
    A, B, D, L2 = _segments(points)
    X = np.asarray(latlon, dtype=np.float64).reshape(-1, 2)  # (N, 2)
    AP = X[:, None, :] - A[None, :, :]                       # (N, S, 2)
    t = np.clip(np.sum(AP * D[None, :, :], axis=2) / L2[None, :], 0.0, 1.0)
    C = A[None, :, :] + t[:, :, None] * D[None, :, :]
    d = np.sqrt(np.sum((X[:, None, :] - C) ** 2, axis=2))    # (N, S)
    idx = np.argmin(d, axis=1)
    return d[np.arange(len(X)), idx], idx


def _cross(latlon, idx, points):
    """z-component of AB x AP on segment idx, with x = lon and y = lat."""
    A, B, D, _ = _segments(points)
    X = np.asarray(latlon, dtype=np.float64).reshape(-1, 2)
    a = A[idx]
    ab_x, ab_y = D[idx, 1], D[idx, 0]              # (lon, lat) components
    ap_x, ap_y = X[:, 1] - a[:, 1], X[:, 0] - a[:, 0]
    return ab_x * ap_y - ab_y * ap_x


# --------------------------------------------------------------------------- #
# Scalar API (unchanged signatures)
# --------------------------------------------------------------------------- #
def distance_from_polyline(point, points):
    """Minimum distance (km) from [lat, lon] to the polyline, and the index of
    the closest segment.  Returns (distance_km, closest_segment_index)."""
    d, idx = _nearest([point], points)
    return float(d[0] * KM_PER_DEG), int(idx[0])


def is_sri_lankan_side(point, points):
    """True if [lat, lon] lies on the far (Sri Lankan / forbidden) side of the
    polyline, i.e. to the left of the nearest segment's direction of travel."""
    _, idx = _nearest([point], points)
    return bool(_cross([point], idx, points)[0] > 0)


# --------------------------------------------------------------------------- #
# Batched API
# --------------------------------------------------------------------------- #
def distance_batch(latlon, points):
    """(dist_km (N,), idx (N,)) for an (N, 2) array of [lat, lon]."""
    d, idx = _nearest(latlon, points)
    return d * KM_PER_DEG, idx


def side_batch(latlon, points):
    """Boolean (N,): True where the fix is on the far side of the polyline."""
    _, idx = _nearest(latlon, points)
    return _cross(latlon, idx, points) > 0


def flags_batch(latlon, points, danger_km):
    """(far_side (N,), inside_buffer (N,), dist_km (N,), seg_idx (N,))."""
    d, idx = _nearest(latlon, points)
    far = _cross(latlon, idx, points) > 0
    d_km = d * KM_PER_DEG
    return far, d_km < danger_km, d_km, idx


def segment_normals(points):
    """(S, 2) unit normals of each segment in (lon, lat) components, pointing
    to the far side (the side where the cross product above is positive)."""
    A, B, D, _ = _segments(points)
    dx, dy = D[:, 1], D[:, 0]
    n = np.stack([-dy, dx], axis=1)
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def zone_of(lat, lon, points, danger_km, caution_km):
    """Exact zone label for one fix: DANGER on the far side or inside the
    danger buffer, CAUTION inside the caution band, otherwise SAFE."""
    far, buf, d, _ = flags_batch([[lat, lon]], points, danger_km)
    if far[0] or buf[0]:
        return "DANGER"
    if d[0] < caution_km:
        return "CAUTION"
    return "SAFE"


def zone_batch(latlon, points, danger_km, caution_km):
    far, buf, d, _ = flags_batch(latlon, points, danger_km)
    out = np.where(far | buf, "DANGER", np.where(d < caution_km, "CAUTION", "SAFE"))
    return out.astype(object)
