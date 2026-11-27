"""Tải và phân tích dữ liệu COSMIC-2 rapid/level2 (scnLv2) cho khu vực Việt Nam.

Đặc điểm scnLv2:
  - Đo che khuất rìa (RO) xuyên tầng F (~60–540 km), biến `occheight` [km].
  - Lọc tầng F (`occheight >= 200 km`) để loại nhiễu Sporadic-E (90–140 km).
  - Lọc chất lượng qua `bitstatus1 == "0"`, `bitstatus2 == "0"` và `RFI`.
"""
import argparse
import io
import sys
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

RAPID_LEVEL2_BASE = "https://data.cosmic.ucar.edu/gnss-ro/cosmic2/rapid/level2"


def scnlv2_url(year: int, doy: int) -> str:
    return f"{RAPID_LEVEL2_BASE}/{year}/{doy:03d}/scnLv2_rapid_{year}_{doy:03d}.tar.gz"


def fetch_scnlv2(year: int, doy: int, out_dir: Path) -> Path:
    """Tải scnLv2_rapid_YYYY_DDD.tar.gz nếu chưa có trên đĩa."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"scnLv2_rapid_{year}_{doy:03d}.tar.gz"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest

    url = scnlv2_url(year, doy)
    tmp = dest.with_suffix(".gz.part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=256 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)
    return dest


def parse_scnlv2_tar(
    tar_path: Path,
    lat_min: float = config.LAT_MIN,
    lat_max: float = config.LAT_MAX,
    lon_min: float = config.LON_MIN,
    lon_max: float = config.LON_MAX,
    min_occheight_km: float = 200.0,
    max_rfi: float | None = 0.05,
    require_clean_bitstatus: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Đọc tar.gz scnLv2, lọc hộp tọa độ + tầng F + RFI/bitstatus.

    Trả về (DataFrame điểm hợp lệ, dict thống kê chẩn đoán của ngày).
    """
    rows = []
    n_total_files = 0
    n_bitstatus_ok = 0
    n_vn_raw_files = 0
    n_vn_f_files = 0
    n_vn_night_files = 0
    n_vn_raw_pts = 0
    n_vn_f_pts = 0
    n_vn_rfi_dropped = 0
    leo_ids = set()

    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            n_total_files += 1
            raw = tf.extractfile(m).read()
            with nc.Dataset("mem.nc", memory=raw) as ds:
                bs1 = str(ds.getncattr("bitstatus1")) if "bitstatus1" in ds.ncattrs() else "0"
                bs2 = str(ds.getncattr("bitstatus2")) if "bitstatus2" in ds.ncattrs() else "0"
                leo = str(ds.getncattr("leoId")) if "leoId" in ds.ncattrs() else "?"
                leo_ids.add(leo)
                if require_clean_bitstatus and (bs1 != "0" or bs2 != "0"):
                    continue
                n_bitstatus_ok += 1

                lat = np.asarray(ds.variables["lat"][:], dtype=float)
                lon = np.asarray(ds.variables["lon"][:], dtype=float)
                lon = np.where(lon > 180.0, lon - 360.0, lon)
                box_mask = (
                    (lat >= lat_min) & (lat <= lat_max) &
                    (lon >= lon_min) & (lon <= lon_max)
                )
                if not np.any(box_mask):
                    continue
                n_vn_raw_files += 1
                n_vn_raw_pts += int(np.sum(box_mask))

                occ = np.asarray(ds.variables["occheight"][:], dtype=float)
                rfi = np.asarray(ds.variables["RFI"][:], dtype=float)
                f_mask = box_mask & np.isfinite(occ) & (occ >= min_occheight_km)
                if not np.any(f_mask):
                    continue
                n_vn_f_files += 1
                n_vn_f_pts += int(np.sum(f_mask))

                if max_rfi is not None:
                    clean_mask = f_mask & np.isfinite(rfi) & (rfi <= max_rfi)
                    n_vn_rfi_dropped += int(np.sum(f_mask) - np.sum(clean_mask))
                else:
                    clean_mask = f_mask
                if not np.any(clean_mask):
                    continue

                t0 = datetime(
                    int(ds.getncattr("year")),
                    int(ds.getncattr("month")),
                    int(ds.getncattr("day")),
                    int(ds.getncattr("hour")),
                    int(ds.getncattr("minute")),
                    int(float(ds.getncattr("second"))),
                    tzinfo=timezone.utc,
                )
                t_sec = np.asarray(ds.variables["time"][:], dtype=float)[clean_mask]
                times = [t0 + timedelta(seconds=float(s)) for s in t_sec]
                hrs = np.array([t.hour for t in times])
                if np.any((hrs >= 12) & (hrs <= 17)):
                    n_vn_night_files += 1

                prn = int(ds.getncattr("occsatId")) if "occsatId" in ds.ncattrs() else -1
                rows.append(pd.DataFrame({
                    "time": times,
                    "utc_hour": hrs,
                    "lat": lat[clean_mask],
                    "lon": lon[clean_mask],
                    "occheight_km": occ[clean_mask],
                    "elevation_deg": np.asarray(ds.variables["elevation"][:], dtype=float)[clean_mask],
                    "s4_L1": np.asarray(ds.variables["s4_L1"][:], dtype=float)[clean_mask],
                    "s4_L2": np.asarray(ds.variables["s4_L2"][:], dtype=float)[clean_mask],
                    "sigma_phi_L1": np.asarray(ds.variables["sigma_phi_L1"][:], dtype=float)[clean_mask],
                    "sigma_phi_L2": np.asarray(ds.variables["sigma_phi_L2"][:], dtype=float)[clean_mask],
                    "slip_L1": np.asarray(ds.variables["slip_L1"][:], dtype=int)[clean_mask],
                    "slip_L2": np.asarray(ds.variables["slip_L2"][:], dtype=int)[clean_mask],
                    "rfi": rfi[clean_mask],
                    "l1_snr": np.asarray(ds.variables["L1_SNR"][:], dtype=float)[clean_mask],
                    "l2_snr": np.asarray(ds.variables["L2_SNR"][:], dtype=float)[clean_mask],
                    "leo_id": leo,
                    "prn_id": prn,
                    "profile_id": m.name,
                }))

    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        # Loại giá trị sentinel (-999)
        for col in ("s4_L1", "s4_L2", "sigma_phi_L1", "sigma_phi_L2"):
            df.loc[df[col] < 0, col] = np.nan

    stats = {
        "tar_mb": round(tar_path.stat().st_size / 1e6, 2),
        "total_files": n_total_files,
        "bitstatus_ok_files": n_bitstatus_ok,
        "leo_ids": sorted(leo_ids),
        "vn_raw_files": n_vn_raw_files,
        "vn_raw_pts": n_vn_raw_pts,
        "vn_f_files": n_vn_f_files,
        "vn_f_pts": n_vn_f_pts,
        "vn_rfi_dropped": n_vn_rfi_dropped,
        "vn_night_files": n_vn_night_files,
        "vn_night_pts": int(((df.utc_hour >= 12) & (df.utc_hour <= 17)).sum()) if not df.empty else 0,
    }
    return df, stats


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Tải & lọc COSMIC-2 scnLv2_rapid cho khung VN")
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "raw" / "cosmic2"))
    args = p.parse_args()

    dt = datetime.strptime(args.date, "%Y-%m-%d")
    doy = dt.timetuple().tm_yday
    tar = fetch_scnlv2(dt.year, doy, Path(args.out))
    df, st = parse_scnlv2_tar(tar)
    print(f"=== scnLv2 {args.date} (DOY {doy:03d}) ===")
    for k, v in st.items():
        print(f"  {k}: {v}")
    if not df.empty:
        night = df[(df.utc_hour >= 12) & (df.utc_hour <= 17)]
        print(f"  night s4_L1 valid={night.s4_L1.notna().sum()}, max={night.s4_L1.max():.3f}, p95={night.s4_L1.quantile(0.95):.3f}")
