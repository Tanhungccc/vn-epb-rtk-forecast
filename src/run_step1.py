"""Bước 1 của pipeline: 1 ngày -> ROTI khu vực VN + chỉ số bão từ.

Dùng:
  python -m src.run_step1 --date 2024-05-10          # bão Gannon
  python -m src.run_step1 --date 2024-06-04          # ngày tĩnh đối chứng
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config  # noqa: E402
from src import fetch_madrigal, roti  # noqa: E402
from src import fetch_swpc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    y, m, d = (int(x) for x in args.date.split("-"))
    raw_tec = ROOT / "data" / "raw" / "tec"
    out = ROOT / "data" / "out"

    # 1. Tải map TEC toàn cầu của ngày
    if args.skip_download:
        path = raw_tec / f"gps{y % 100:02d}{m:02d}{d:02d}.hdf5"
        if not path.exists():
            raise FileNotFoundError(path)
    else:
        path = fetch_madrigal.fetch_day(y, m, d, raw_tec)

    # 2. Tính ROT/ROTI + thống kê VN
    df = fetch_madrigal.load_gridded_tec(path)
    print(f"{path.name}: {len(df):,} rows, {df.time.nunique()} maps")
    df_roti = roti.compute_roti(roti.compute_rot(df))
    summary = roti.vn_summary(df_roti)
    hourly = roti.hourly_stats(summary)

    summary.to_parquet(out / f"{path.stem}_vn_5min.parquet")
    hourly.to_parquet(out / f"{path.stem}_vn_hourly.parquet")

    # 3. Chỉ số bão từ (Kp 7 ngày + Bz 24h gần nhất)
    try:
        stats = fetch_swpc.save_snapshot(ROOT / "data" / "raw" / "swpc")
        print(f"SWPC: Kp now={stats['kp_now']:.1f} | max 7d={stats['kp_max_7d']:.1f} | "
              f"Bz now={stats['bz_now']} | Bz min 24h={stats['bz_min_24h']}")
    except Exception as e:  # noqa: BLE001
        print(f"SWPC fetch failed (không chặn pipeline): {e}")

    # 4. Báo cáo ngày
    print(f"\n===== {args.date} =====")
    print(hourly.to_string(index=False))
    epb_total = int(summary.n_epb_cells.sum())
    night = summary[(summary.utc_hour >= 12) & (summary.utc_hour <= 17)]
    print(f"\nTổng ô EPB-flag cả ngày: {epb_total} | trong khung tối VN (12-17 UTC): {int(night.n_epb_cells.sum())}")
    print(f"ROTI max ngày: {summary.roti_max.max():.2f} TECU/phút (ngưỡng {config.ROTI_THRESHOLD})")
    print(f"Output: {out}\\{path.stem}_vn_*.parquet")


if __name__ == "__main__":
    main()
