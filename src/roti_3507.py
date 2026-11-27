"""Đọc, lọc hộp VN và chuẩn hóa Madrigal kindat 3507 (ROTI chuẩn tính từ LOS 30s).

Lưu ý đơn vị trên Madrigal kindat 3507:
  - Cột `roti` có đơn vị `TECu/sec`.
  - Quy đổi sang `TECU/phút`: `roti_tecu_min = roti * 60.0` để so sánh trực tiếp với ROTI-proxy.
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def load_roti_3507_region(
    h5_path: Path,
    lat_min: float = config.LAT_MIN,
    lat_max: float = config.LAT_MAX,
    lon_min: float = 99.0,
    lon_max: float = 122.0,
    chunk_size: int = 2_000_000,
) -> pd.DataFrame:
    """Đọc theo chunk file roti_DDD_YYYY.h5 (~657 MB) và lọc ngay theo hộp tọa độ."""
    parts = []
    with h5py.File(h5_path, "r") as f:
        ds = f["Data/Table Layout"]
        n = len(ds)
        cols = ["year", "month", "day", "hour", "min", "sec", "gdlat", "glon", "roti", "elm"]
        for start in range(0, n, chunk_size):
            tbl = ds[start : start + chunk_size]
            lat = tbl["gdlat"]
            lon = tbl["glon"]
            mask = (
                np.isfinite(lat) & np.isfinite(lon) & np.isfinite(tbl["roti"]) &
                (lat >= lat_min) & (lat <= lat_max) &
                (lon >= lon_min) & (lon <= lon_max)
            )
            if not np.any(mask):
                continue
            sub = tbl[mask]
            df_c = pd.DataFrame({c: sub[c] for c in cols})
            for sc in ("gps_site", "gnss_type"):
                if sc in sub.dtype.names:
                    df_c[sc] = [
                        v.decode("ascii", "ignore").strip() if isinstance(v, (bytes, np.bytes_)) else str(v).strip()
                        for v in sub[sc]
                    ]
            parts.append(df_c)

    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["time"] = pd.to_datetime(
        dict(year=df.year, month=df.month, day=df.day, hour=df.hour, minute=df["min"], second=df.sec),
        utc=True,
    )
    df["roti_tecu_min"] = df["roti"].astype(float) * 60.0
    df["utc_hour"] = df["time"].dt.hour
    df["lat_bin"] = np.round(df["gdlat"]).astype(int)
    df["lon_bin"] = np.round(df["glon"]).astype(int)
    df["time_5min"] = df["time"].dt.floor("5min")
    return df
