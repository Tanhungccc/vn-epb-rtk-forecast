"""Tải gói podTc2 NRT hàng ngày của COSMIC-2 (TEC tuyệt đối + S4 on-board).

URL: {base}/{year}/{ddd}/podTc2_nrt_{year}_{ddd}.tar.gz (~500 MB/ngày)
Bên trong: podTc2_C2E{n}.YYYY.DDD.HH.MM...nc (NetCDF4, 1 file/vệ tinh-GNSS pair).
"""
import sys
import tarfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def day_url(year: int, doy: int) -> str:
    return f"{config.COSMIC2_LEVEL1B}/{year}/{doy:03d}/podTc2_nrt_{year}_{doy:03d}.tar.gz"


def fetch_day(year: int, doy: int, out_dir: Path, extract: bool = True) -> Path:
    url = day_url(year, doy)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = out_dir / f"podTc2_nrt_{year}_{doy:03d}.tar.gz"

    if not (tar_path.exists() and tar_path.stat().st_size > 400_000_000):
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            with open(tar_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {done/1e6:.0f}/{total/1e6:.0f} MB", end="")
        print()
    else:
        print(f"  {tar_path.name} exists, skip download")

    if extract:
        extract_dir = out_dir / f"{year}_{doy:03d}"
        extract_dir.mkdir(exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(extract_dir)  # noqa: S202 - nguồn chính thức UCAR
        print(f"  extracted -> {extract_dir}")
        return extract_dir
    return tar_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Tải podTc2 NRT COSMIC-2 của 1 ngày (YYYY DDD)")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--doy", type=int, required=True, help="Day of year")
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "raw" / "cosmic2"))
    p.add_argument("--no-extract", action="store_true")
    args = p.parse_args()

    res = fetch_day(args.year, args.doy, Path(args.out), extract=not args.no_extract)
    print("OK:", res)
