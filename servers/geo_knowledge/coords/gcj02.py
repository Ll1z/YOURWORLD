"""GCJ-02 与 WGS84 互转（无第三方依赖，仅 numpy）。"""

import numpy as np

A = 6378245.0
EE = 0.00669342162296594323


def _out_of_china(lon, lat):
    return (lon < 72.004) | (lon > 137.8347) | (lat < 0.8293) | (lat > 55.8271)


def _transform_lat(x, y):
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * np.sqrt(np.abs(x))
    ret += (20.0 * np.sin(6.0 * x * np.pi) + 20.0 * np.sin(2.0 * x * np.pi)) * 2.0 / 3.0
    ret += (20.0 * np.sin(y * np.pi) + 40.0 * np.sin(y / 3.0 * np.pi)) * 2.0 / 3.0
    ret += (160.0 * np.sin(y / 12.0 * np.pi) + 320.0 * np.sin(y * np.pi / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lon(x, y):
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * np.sqrt(np.abs(x))
    ret += (20.0 * np.sin(6.0 * x * np.pi) + 20.0 * np.sin(2.0 * x * np.pi)) * 2.0 / 3.0
    ret += (20.0 * np.sin(x * np.pi) + 40.0 * np.sin(x / 3.0 * np.pi)) * 2.0 / 3.0
    ret += (150.0 * np.sin(x / 12.0 * np.pi) + 300.0 * np.sin(x / 30.0 * np.pi)) * 2.0 / 3.0
    return ret


def wgs84_to_gcj02(lon, lat):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    dlat = _transform_lat(lon - 105.0, lat - 35.0)
    dlon = _transform_lon(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * np.pi
    magic = 1.0 - EE * np.sin(radlat) ** 2
    sqrtmagic = np.sqrt(magic)
    dlat = (dlat * 180.0) / ((A * (1 - EE)) / (magic * sqrtmagic) * np.pi)
    dlon = (dlon * 180.0) / (A / sqrtmagic * np.cos(radlat) * np.pi)
    glon, glat = lon + dlon, lat + dlat
    mask = _out_of_china(lon, lat)
    return np.where(mask, lon, glon), np.where(mask, lat, glat)


def gcj02_to_wgs84(lon, lat, iters=30, tol=1e-12):
    """迭代反解：GCJ-02 -> WGS84。"""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    wlon, wlat = lon.copy(), lat.copy()
    for _ in range(iters):
        glon, glat = wgs84_to_gcj02(wlon, wlat)
        dlon, dlat = glon - lon, glat - lat
        wlon, wlat = wlon - dlon, wlat - dlat
        if np.max(np.hypot(dlon, dlat)) < tol:
            break
    return wlon, wlat