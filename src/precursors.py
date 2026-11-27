"""Trích xuất các tiền tố vật lý cho tầm dự báo 3–6 giờ:
  1. NASA OMNI2 hourly (zero-auth): F10.7, Bz_GSM, v_sw, Ey = -v_sw*Bz [mV/m], Dst, Kp.
  2. ESA GIM IONEX (zero-auth, ~1 MB/ngày): bản đồ TEC phủ kín 100%, tính độ chênh EIA Bắc–Nam
     buổi chiều (07–10 UTC = 14h–17h LT tại Việt Nam).
  3. Tuần hoàn mùa: sin(DOY), cos(DOY).
"""
import gzip
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OMNI2_BASE = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni"
ESA_IONEX_BASE = "http://navigation-office.esa.int/products/gnss-products"
GPS_EPOCH = date(1980, 1, 6)


def gps_week_of(d: date) -> int:
    return (d - GPS_EPOCH).days // 7


def fetch_omni2_year(year: int, out_dir: Path) -> Path:
    """Tải omni2_YYYY.dat (~3.5 MB/năm) nếu chưa có hoặc đã cũ >1 ngày với năm hiện tại."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"omni2_{year}.dat"
    if dest.exists() and dest.stat().st_size > 100_000:
        return dest
    url = f"{OMNI2_BASE}/omni2_{year}.dat"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def load_omni2_hourly(dat_path: Path) -> pd.DataFrame:
    """Đọc các cột chuẩn OMNI2 hourly (1-based Fortran cols -> 0-based Python):
    0: Year, 1: DOY, 2: Hour, 16: Bz_GSM (nT, fill 999.9),
    24: V_sw (km/s, fill 9999.), 38: Kp*10 (fill 99), 40: Dst (nT, fill 99999),
    50: F10.7 (sfu, fill 999.9).
    """
    usecols = [0, 1, 2, 16, 24, 38, 40, 50]
    names = ["year", "doy", "hour", "bz_gsm", "v_sw", "kp10", "dst", "f107"]
    df = pd.read_csv(dat_path, sep=r"\s+", header=None, usecols=usecols, names=names)
    df["time"] = (
        pd.to_datetime(df["year"].astype(str) + df["doy"].astype(str).str.zfill(3), format="%Y%j", utc=True)
        + pd.to_timedelta(df["hour"], unit="h")
    )
    df.loc[df["bz_gsm"] >= 999.0, "bz_gsm"] = np.nan
    df.loc[df["v_sw"] >= 9999.0, "v_sw"] = np.nan
    df.loc[df["kp10"] >= 99, "kp10"] = np.nan
    df.loc[df["dst"] >= 99999, "dst"] = np.nan
    df.loc[df["f107"] >= 999.0, "f107"] = np.nan

    df["kp"] = df["kp10"] / 10.0
    # Ey = -v_sw [km/s] * Bz [nT] * 1e-3  -> [mV/m]
    df["ey_mvm"] = -df["v_sw"] * df["bz_gsm"] * 1e-3
    df["sin_doy"] = np.sin(2.0 * math.pi * df["doy"] / 365.25)
    df["cos_doy"] = np.cos(2.0 * math.pi * df["doy"] / 365.25)
    return df[["time", "year", "doy", "hour", "f107", "bz_gsm", "v_sw", "ey_mvm", "kp", "dst", "sin_doy", "cos_doy"]]


def fetch_esa_ionex(d: date, out_dir: Path) -> Path:
    """Tải GIM IONEX của ESA (ưu tiên bản 1h rapid, fallback 2h final)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wk = gps_week_of(d)
    doy = d.timetuple().tm_yday
    cands = [
        f"ESA0OPSRAP_{d.year}{doy:03d}0000_01D_01H_GIM.INX.gz",
        f"ESA0OPSFIN_{d.year}{doy:03d}0000_01D_02H_GIM.INX.gz",
    ]
    for fname in cands:
        dest = out_dir / fname
        if dest.exists() and dest.stat().st_size > 10_000:
            return dest
        url = f"{ESA_IONEX_BASE}/{wk}/{fname}"
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            dest.write_bytes(r.content)
            return dest
    raise RuntimeError(f"No ESA IONEX found for {d} (week {wk}, doy {doy:03d})")


def parse_ionex_eia_gradient(inx_gz_path: Path) -> pd.DataFrame:
    """Đọc GIM IONEX (.INX.gz), trích xuất TEC vùng Bắc Bộ (17.5–22.5N) và Nam Bộ (7.5–12.5N)
    trên dải kinh độ 100°–110°E cho mỗi epoch UTC.
    """
    raw = gzip.decompress(inx_gz_path.read_bytes()).decode("ascii", errors="ignore")
    lines = raw.splitlines()
    exponent = -1
    lats, lons, maps, epochs = None, None, [], []

    i = 0
    while i < len(lines):
        ln = lines[i]
        if "EXPONENT" in ln[60:]:
            exponent = int(ln[:6].strip())
        elif "LON1 / LON2 / DLON" in ln[60:]:
            p = ln[:60].split()
            lon1, lon2, dlon = float(p[0]), float(p[1]), float(p[2])
            lons = np.arange(lon1, lon2 + 0.1 * dlon, dlon)
        elif "START OF TEC MAP" in ln[60:]:
            i += 1
            ep_line = lines[i]
            p = ep_line[:60].split()
            t = datetime(int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4]), int(float(p[5])), tzinfo=timezone.utc)
            i += 1
            grid_rows = []
            lat_vals = []
            while i < len(lines) and "END OF TEC MAP" not in lines[i][60:]:
                if "LAT/LON1/LON2/DLON/H" in lines[i][60:]:
                    lat_v = float(lines[i][:8].strip())
                    lat_vals.append(lat_v)
                    vals = []
                    i += 1
                    while len(vals) < len(lons) and i < len(lines):
                        vals.extend(int(x) for x in lines[i].split())
                        i += 1
                    grid_rows.append(np.array(vals[: len(lons)], dtype=float) * (10.0 ** exponent))
                else:
                    i += 1
            if lats is None:
                lats = np.array(lat_vals)
            epochs.append(t)
            maps.append(np.vstack(grid_rows))
        i += 1

    lats_arr = np.array(lats)
    lon_mask = (lons >= 100.0) & (lons <= 110.0)
    north_mask = (lats_arr >= 17.5) & (lats_arr <= 22.5)
    south_mask = (lats_arr >= 7.5) & (lats_arr <= 12.5)

    out_rows = []
    for t, mat in zip(epochs, maps):
        n_tec = float(np.nanmean(mat[np.ix_(north_mask, lon_mask)]))
        s_tec = float(np.nanmean(mat[np.ix_(south_mask, lon_mask)]))
        out_rows.append({
            "time": t,
            "utc_hour": t.hour,
            "tec_eia_north": n_tec,
            "tec_trough_south": s_tec,
            "eia_diff_ns": n_tec - s_tec,
            "eia_ratio_ns": n_tec / s_tec if s_tec > 0 else np.nan,
        })
    df = pd.DataFrame(out_rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


if __name__ == "__main__":
    raw_omni = Path("data/raw/omni")
    raw_inx = Path("data/raw/ionex")
    omni_path = fetch_omni2_year(2024, raw_omni)
    df_omni = load_omni2_hourly(omni_path)

    for ds in ("2024-05-10", "2024-05-11", "2024-06-04"):
        d = datetime.strptime(ds, "%Y-%m-%d").date()
        inx_path = fetch_esa_ionex(d, raw_inx)
        df_inx = parse_ionex_eia_gradient(inx_path)
        aft = df_inx[(df_inx.utc_hour >= 7) & (df_inx.utc_hour <= 10)]  # 14-17h LT
        om_day = df_omni[df_omni.time.dt.date == d]
        print(
            f"{ds} | F10.7={om_day.f107.mean():.1f}, Dst_min={om_day.dst.min():.0f} nT, "
            f"Kp_max={om_day.kp.max():.1f}, Ey_max={om_day.ey_mvm.max():.2f} mV/m | "
            f"Afternoon EIA (14-17h LT): N={aft.tec_eia_north.mean():.1f}, "
            f"S={aft.tec_trough_south.mean():.1f}, diff={aft.eia_diff_ns.mean():.1f} TECU "
            f"({inx_path.name})"
        )
