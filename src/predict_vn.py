"""CLI suy luận nhanh: xuất bảng cảnh báo EPB 3 miền + toàn VN (+1h..+6h) cho 1 ngày bất kỳ.

Dùng:
  .venv/Scripts/python.exe -m src.predict_vn --date 2026-03-04 --hour 14
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import train_models  # noqa: E402


def risk_level(prob: float, thresh: float) -> str:
    if prob >= thresh:
        return "DO (Nguy co mat khoa RTK)"
    if prob >= 0.30:
        return "VANG (Nhieu vua)"
    return "XANH (Binh thuong)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD trong dataset")
    ap.add_argument("--hour", type=int, default=14, help="Gio muc tieu UTC (12..17)")
    args = ap.parse_args()

    df = pd.read_parquet(ROOT / "data" / "dataset" / "hourly_features.parquet")
    day_df = df[df["date"] == args.date]
    if day_df.empty:
        raise ValueError(f"Khong tim thay ngay {args.date} trong hourly_features.parquet")

    print(f"=== Du bao EPB (S4 >= 0.50) cho {args.date} {args.hour:02d}:00 UTC ({(args.hour+7)%24:02d}:00 gio VN) ===")
    print(f"{'Vung':8s} | {'Lead':>4s} | {'Model':10s} | {'Prob':>6s} | {'Th':>4s} | Trang thai")
    print("-" * 72)
    for tgt in ("vn", "north", "central", "south"):
        for lead in (1, 3, 6):
            p = ROOT / "models" / f"{tgt}_lead{lead}h.joblib"
            if not p.exists():
                continue
            b = joblib.load(p)
            _, s4_col = train_models.COL_MAP[tgt]
            work, _ = train_models.build_lagged(day_df, lead, s4_col)
            row = work[work["utc_hour"] == args.hour]
            if row.empty:
                continue
            prob = float(b["calibrated_model"].predict_proba(row[b["feat_cols"]])[:, 1][0])
            th = float(b["threshold"])
            print(f"{tgt:8s} | {lead:>3d}h | {b['model_name']:10s} | {prob:6.3f} | {th:4.2f} | {risk_level(prob, th)}")


if __name__ == "__main__":
    main()
