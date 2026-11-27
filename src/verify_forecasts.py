"""Đối chiếu tự động nhật ký dự báo Live (data/forecast_log.parquet) với vệ tinh COSMIC-2 thật.

Chạy mỗi sáng (sau 24–48h khi UCAR cập nhật scnLv2_rapid):
  python -m src.verify_forecasts

Tính các chỉ số vận hành thực tế:
  - Hit Rate (POD), False Alarm Rate (FAR), Brier Score -> data/verification_log.parquet.
  - Xuất bản ghi đầy đủ đặc trưng của từng ngày đã đối chiếu vào
    data/dataset/live_accumulated.parquet (khớp schema hourly_features.parquet, chống trùng ngày).
"""
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import precursors, scnlv2  # noqa: E402

FORECAST_LOG = ROOT / "data" / "forecast_log.parquet"
VERIFY_LOG = ROOT / "data" / "verification_log.parquet"
ACCUM_PATH = ROOT / "data" / "dataset" / "live_accumulated.parquet"

BAND_BOUNDS = {
    "vn": (6.0, 24.0, True),
    "north": (18.0, 24.0, True),
    "central": (12.0, 18.0, False),
    "south": (6.0, 12.0, False),
}


def _c2_band_stats(df_c2: pd.DataFrame, target: str, utc_hour: int) -> tuple[int, float, float]:
    if df_c2.empty or target not in BAND_BOUNDS:
        return 0, float("nan"), float("nan")
    lo, hi, inc = BAND_BOUNDS[target]
    sub = df_c2[(df_c2["utc_hour"] == utc_hour) & df_c2["lon"].between(101.0, 116.0)]
    if sub.empty:
        return 0, float("nan"), float("nan")
    m = (sub["lat"] >= lo) & (sub["lat"] <= hi if inc else sub["lat"] < hi)
    band = sub[m]
    if band.empty:
        return 0, float("nan"), float("nan")
    return int(len(band)), float(band["s4_L1"].quantile(0.95)), float(band["s4_L1"].max())


def _obs_s4_for_target(df_c2: pd.DataFrame, target: str, utc_hour: int) -> float:
    _, _, s4_max = _c2_band_stats(df_c2, target, utc_hour)
    return s4_max


def _build_daily_accum_rows(d, day_fc: pd.DataFrame, df_c2: pd.DataFrame, raw_inx: Path) -> list[dict]:
    """Tạo các dòng đặc trưng chuẩn schema hourly_features.parquet cho ngày d."""
    doy = d.timetuple().tm_yday
    sin_doy = math.sin(2.0 * math.pi * doy / 365.25)
    cos_doy = math.cos(2.0 * math.pi * doy / 365.25)

    # Thử lấy ESA IONEX của ngày d
    inx_h = pd.DataFrame()
    try:
        inx_p = precursors.fetch_esa_ionex(d, raw_inx)
        df_inx = precursors.parse_ionex_eia_gradient(inx_p)
        inx_h = df_inx.groupby("utc_hour")[["tec_eia_north", "tec_trough_south", "eia_diff_ns", "eia_ratio_ns"]].mean()
        inx_h = inx_h.reindex(range(0, 24)).interpolate(method="linear", limit_direction="both")
    except Exception:
        pass

    # Tổng hợp SWPC trung bình theo ngày từ forecast_log của ngày d
    swpc_cols = ["f107", "bz_gsm", "v_sw", "ey_mvm", "kp", "dst"]
    sw_mean = {c: float(day_fc[c].mean()) if c in day_fc.columns and day_fc[c].notna().any() else float("nan") for c in swpc_cols}

    rows = []
    for hr in range(6, 18):
        c2_h = df_c2[df_c2["utc_hour"] == hr] if not df_c2.empty else pd.DataFrame()
        reg_pts = int(len(c2_h))
        reg_p95 = float(c2_h["s4_L1"].quantile(0.95)) if reg_pts > 0 else float("nan")
        reg_max = float(c2_h["s4_L1"].max()) if reg_pts > 0 else float("nan")

        vn_pts, vn_p95, vn_max = _c2_band_stats(df_c2, "vn", hr)
        n_pts, n_p95, n_max = _c2_band_stats(df_c2, "north", hr)
        c_pts, c_p95, c_max = _c2_band_stats(df_c2, "central", hr)
        s_pts, s_p95, s_max = _c2_band_stats(df_c2, "south", hr)

        inx_r = inx_h.loc[hr] if (not inx_h.empty and hr in inx_h.index) else None
        rows.append({
            "date": d.isoformat(),
            "year": d.year,
            "month": d.month,
            "doy": doy,
            "utc_hour": hr,
            "lt_hour": (hr + 7) % 24,
            "sin_doy": sin_doy,
            "cos_doy": cos_doy,
            **sw_mean,
            "eia_north": float(inx_r["tec_eia_north"]) if inx_r is not None else float("nan"),
            "eia_south": float(inx_r["tec_trough_south"]) if inx_r is not None else float("nan"),
            "eia_diff": float(inx_r["eia_diff_ns"]) if inx_r is not None else float("nan"),
            "eia_ratio": float(inx_r["eia_ratio_ns"]) if inx_r is not None else float("nan"),
            "c2_reg_pts": reg_pts,
            "c2_reg_s4_p95": reg_p95,
            "c2_reg_s4_max": reg_max,
            "c2_vn_pts": vn_pts,
            "c2_vn_s4_p95": vn_p95,
            "c2_vn_s4_max": vn_max,
            "c2_north_pts": n_pts,
            "c2_north_s4_p95": n_p95,
            "c2_north_s4_max": n_max,
            "c2_north_epb": int(n_max >= 0.30) if np.isfinite(n_max) else np.nan,
            "c2_central_pts": c_pts,
            "c2_central_s4_p95": c_p95,
            "c2_central_s4_max": c_max,
            "c2_central_epb": int(c_max >= 0.30) if np.isfinite(c_max) else np.nan,
            "c2_south_pts": s_pts,
            "c2_south_s4_p95": s_p95,
            "c2_south_s4_max": s_max,
            "c2_south_epb": int(s_max >= 0.30) if np.isfinite(s_max) else np.nan,
            "igs_hksl_roti": float("nan"),
            "igs_cusv_roti": float("nan"),
            "igs_pimo_roti": float("nan"),
            "igs_ntus_roti": float("nan"),
        })
    return rows


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

    past_dates = sorted(d for d in df_fc["fc_date"].dropna().unique() if d < today_utc)
    if not past_dates:
        print(f"Chưa có ngày quá khứ nào trong forecast_log.parquet để đối chiếu (mới nhất: {today_utc}).")
        return pd.DataFrame()

    raw_c2 = ROOT / "data" / "raw" / "cosmic2"
    raw_inx = ROOT / "data" / "raw" / "ionex"
    matched_rows = []
    accum_rows = []

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
        accum_rows.extend(_build_daily_accum_rows(d, day_fc, df_c2, raw_inx))

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

    if accum_rows:
        df_acc_new = pd.DataFrame(accum_rows)
        ACCUM_PATH.parent.mkdir(parents=True, exist_ok=True)
        if ACCUM_PATH.exists():
            df_acc_old = pd.read_parquet(ACCUM_PATH)
            df_acc = pd.concat([df_acc_old, df_acc_new], ignore_index=True)
        else:
            df_acc = df_acc_new
        df_acc = df_acc.drop_duplicates(subset=["date", "utc_hour"], keep="last").sort_values(["date", "utc_hour"]).reset_index(drop=True)
        df_acc.to_parquet(ACCUM_PATH, index=False)
        print(f"Đã cập nhật {ACCUM_PATH.name}: {df_acc['date'].nunique()} ngày ({len(df_acc)} giờ).")

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
