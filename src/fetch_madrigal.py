"""Tải bản đồ TEC toàn cầu 1°x1°x5min từ CEDAR Madrigal (kinst 8000, kindat 3500).

File HDF5 mỗi ngày ~150 MB. Server bỏ qua Range header -> tải 1 lần,
nếu đứt giữa chừng thì xóa file và tải lại (pipeline tự retry).
"""
import sys
import time
import urllib.request
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def madrigal_tec_url(year: int, month: int, day: int, kindat: int = config.MADRIGAL_KINDAT) -> str:
    """FTP-interface dạng năm: /kinst/8000/year/YYYY/kindat/KINDAT/format/hdf5/ rồi lọc client-side."""
    base = f"{config.MADRIGAL_BASE}/kinst/{config.MADRIGAL_KINST}/year/{year}"
    return f"{base}/kindat/{kindat}/format/hdf5/"


def list_day_files(year: int, month: int, day: int, kindat: int = config.MADRIGAL_KINDAT):
    """Liệt kê link fullFilename của ngày Y-M-D cho kindat 3500 (TEC) hoặc 3507 (ROTI)."""
    import re
    import urllib.parse
    from datetime import date

    import requests

    r = requests.get(madrigal_tec_url(year, month, day, kindat=kindat), timeout=120)
    r.raise_for_status()
    hrefs = re.findall(r'href="([^"]+fullFilename[^"]+)"', r.text)
    if kindat == 3507:
        doy = date(year, month, day).timetuple().tm_yday
        tag = f"roti_{doy:03d}_{year}"
    else:
        tag = f"gps{year % 100:02d}{month:02d}{day:02d}g"
    picked = [h for h in set(hrefs) if tag in urllib.parse.unquote(h)]
    return sorted(set(picked))


def download(url: str, dest: Path, timeout: int = 60, max_attempts: int = 40) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp, open(dest, "wb") as out:
                while True:
                    chunk = resp.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    out.flush()
            size = dest.stat().st_size
            print(f"  downloaded {dest.name}: {size/1e6:.1f} MB ({attempts} attempt(s))")
            return dest
        except Exception as e:  # noqa: BLE001
            print(f"  attempt {attempts} error: {e}")
            if dest.exists():
                dest.unlink()  # server ignores Range -> restart clean
            time.sleep(3)
    raise RuntimeError(f"Cannot download {url} after {max_attempts} attempts")


def fetch_day(year: int, month: int, day: int, out_dir: Path, kindat: int = config.MADRIGAL_KINDAT) -> Path:
    """Tải file TEC (3500) hoặc ROTI (3507) của 1 ngày, trả về path file hdf5 đã chọn."""
    from datetime import date

    links = list_day_files(year, month, day, kindat=kindat)
    if not links:
        raise RuntimeError(f"No Madrigal kindat={kindat} file for {year}-{month:02d}-{day:02d}")
    url = "https://cedar.openmadrigal.org" + links[-1]
    if kindat == 3507:
        doy = date(year, month, day).timetuple().tm_yday
        dest = out_dir / f"roti_{doy:03d}_{year}.h5"
    else:
        dest = out_dir / f"gps{year % 100:02d}{month:02d}{day:02d}.hdf5"
    if dest.exists() and dest.stat().st_size > 100_000_000:
        print(f"  {dest.name} exists, skip download")
        return dest
    if dest.exists():
        dest.unlink()
    return download(url, dest)


def load_gridded_tec(path: Path):
    """Đọc hdf5 Madrigal -> bảng (time, gdlat, glon, tec) dạng DataFrame thưa."""
    import pandas as pd

    with h5py.File(path, "r") as f:
        tbl = f["Data/Table Layout"][()]
    df = pd.DataFrame(tbl[["year", "month", "day", "hour", "min", "gdlat", "glon", "tec"]])
    df["time"] = pd.to_datetime(dict(year=df.year, month=df.month, day=df.day,
                                     hour=df.hour, minute=df["min"]), utc=True)
    df = df[["time", "gdlat", "glon", "tec"]].sort_values(["time", "gdlat", "glon"])
    return df


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Tải bản đồ TEC Madrigal của 1 ngày")
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "raw" / "tec"))
    args = p.parse_args()

    y, m, d = (int(x) for x in args.date.split("-"))
    out_dir = Path(args.out)
    files = list_day_files(y, m, d)
    print(f"{args.date}: {len(files)} file(s) available")
    path = fetch_day(y, m, d, out_dir)
    df = load_gridded_tec(path)
    vn = df[(df.gdlat >= config.LAT_MIN) & (df.gdlat <= config.LAT_MAX) &
            (df.glon >= config.LON_MIN) & (df.glon <= config.LON_MAX)]
    print(f"Loaded {len(df):,} grid points | VN box: {len(vn):,} points | {df.time.nunique()} 5-min maps")
