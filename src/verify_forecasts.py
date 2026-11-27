"""Đối chiếu tự động nhật ký dự báo Live (data/forecast_log.parquet) với vệ tinh COSMIC-2 thật.

Chạy mỗi sáng (sau 24–48h khi UCAR cập nhật scnLv2_rapid):
  python -m src.verify_forecasts

Tính các chỉ số vận hành thực tế:
  - Hit Rate (POD): Tỷ lệ bắt trúng các giờ có nhiễu mạnh (S4 >= 0.50).
  - False Alarm Rate (FAR): Tỷ lệ báo động nhầm.
  - Brier Score: Sai số bình phương xác suất (càng nhỏ càng tốt).
Lưu kết quả chi tiết vào data/verification_log.parquet.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import scnlv2  # noqa: E402

FORECAST_LOG = ROOT / "data" / "forecast_log.parquet"
VERIFY_LOG = ROOT / "data" / "verification_log.parquet"

BAND_BOUNDS = {
    "vn": (6.0, 24.0, True),
    "north": (18.0, 24.0, True),
    "central": (12.0, 18.0, False),
    "south": (6.0, 12.0, False),
}


def _obs_s4_for_target(df_c2: pd.DataFrame, target: str, utc_hour: int) -> float:
    if df_c2.empty or target not in BAND_BOUNDS:
        return float("nan")
    lo, hi, inc = BAND_BOUNDS[target]
    sub = df_c2[(df_c2["utc_hour"] == utc_hour) & df_c2["lon"].between(101.0, 116.0)]
    if sub.empty:
        return float("nan")
    m = (sub["lat"] >= lo) & (sub["lat"] <= hi if inc else sub["lat"] < hi)
    band = sub[m]
    if band.empty:
        return float("nan")
    return float(band["s4_L1"].max())


def verify_forecasts(s4_thresh: float = 0.50) -> pd.DataFrame:
    if not FORECAST_LOG.exists():
        print("Chưa có data/forecast_log.parquet. Hãy chạy live_forecast.py trước.")
        return pd.DataFrame()

    df_fc = pd.read_parquet(FORECAST_LOG)
    if df_fc.empty:
        print("forecast_log.parquet đang rỗng.")
        return pd.DataFrame()

    df_fc["run_dt"] = pd.to_datetime(df_fc["run_time_utc"], utc=True, errors="coerce")
    df_fc["fc_date"] = df_fc["run_dt"].dt.date
    today_utc = datetime.now(timezone.utc).date()

    # Chỉ đối chiếu các ngày đã qua (COSMIC-2 rapid thường có sau 24-48h)
    past_dates = sorted(d for d in df_fc["fc_date"].dropna().unique() if d < today_utc)
    if not past_dates:
        print(f"Chưa có ngày quá khứ nào trong forecast_log.parquet để đối chiếu (mới nhất: {today_utc}).")
        return pd.DataFrame()

    raw_c2 = ROOT / "data" / "raw" / "cosmic2"
    matched_rows = []
    for d in past_dates:
        doy = d.timetuple().tm_yday
        try:
            tar_p = scnlv2.fetch_scnlv2(d.year, doy, raw_c2)
            df_c2, _ = scnlv2.parse_scnlv2_tar(
                tar_p, lat_min=6.0, lat_max=24.0, lon_min=101.0, lon_max=116.0, min_occheight_km=200.0
            )
        except Exception as e:  # noqa: BLE001
            print(f"  Bỏ qua {d}: chưa tải được COSMIC-2 ({e})")
            continue

        day_fc = df_fc[df_fc["fc_date"] == d]
        for _, r in day_fc.iterrows():
            s4_obs = _obs_s4_for_target(df_c2, str(r["target"]), int(r["utc_hour"]))
            if not np.isfinite(s4_obs):
                continue
            y_true = int(s4_obs >= s4_thresh)
            y_pred = int(float(r["prob"]) >= float(r["threshold"]))
            matched_rows.append({
                "date": d.isoformat(),
                "run_hour_utc": r["run_hour_utc"],
                "target": r["target"],
                "lt_label": r["lt_label"],
                "lead": int(r["lead"]),
                "prob": float(r["prob"]),
                "threshold": float(r["threshold"]),
                "s4_obs_max": s4_obs,
                "y_true": y_true,
                "y_pred": y_pred,
            })

    if not matched_rows:
        print("Chưa có cặp (dự báo, COSMIC-2) nào khớp.")
        return pd.DataFrame()

    df_v = pd.DataFrame(matched_rows).drop_duplicates(
        subset=["run_hour_utc", "target", "lead", "lt_label"], keep="last"
    )
    VERIFY_LOG.parent.mkdir(parents=True, exist_ok=True)
    df_v.to_parquet(VERIFY_LOG, index=False)

    tp = int(((df_v["y_true"] == 1) & (df_v["y_pred"] == 1)).sum())
    fn = int(((df_v["y_true"] == 1) & (df_v["y_pred"] == 0)).sum())
    fp = int(((df_v["y_true"] == 0) & (df_v["y_pred"] == 1)).sum())
    hit_rate = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    far = fp / (tp + fp) if (tp + fp) > 0 else float("nan")
    brier = float(np.mean((df_v["prob"] - df_v["y_true"]) ** 2))
    print(f"Đối chiếu hoàn tất: {len(df_v)} mẫu | Hit Rate={hit_rate:.3f} | FAR={far:.3f} | Brier={brier:.4f}")
    return df_v


if __name__ == "__main__":
    verify_forecasts()
