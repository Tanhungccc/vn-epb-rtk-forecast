"""Tính ROT và chỉ số ROTI-proxy từ bản đồ TEC 5 phút của Madrigal + thống kê khu vực VN.

Phương pháp (proxy chuẩn cho ROTI khi chỉ có bản đồ 5 phút):
  ROT  = (TEC_t - TEC_{t-5min}) / 5min  [TECU/phút] trên từng ô lưới 1°
  ROTI = std(ROT) trên cửa sổ trượt ROTI_WINDOW_MAPS (mặc định 5 maps = 20 phút)
  Ô lưới có ROTI >= ROTI_THRESHOLD (0.5 TECU/phút) được coi là dấu hiệu
  bất thường điện ly / bong bóng plasma (EPB proxy).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def compute_rot(df: pd.DataFrame) -> pd.DataFrame:
    """df: (time, gdlat, glon, tec). Thêm cột rot (TECU/phút) so với map trước."""
    df = df.sort_values(["gdlat", "glon", "time"]).copy()
    prev_tec = df.groupby(["gdlat", "glon"])["tec"].shift(1)
    prev_time = df.groupby(["gdlat", "glon"])["time"].shift(1)
    dt_min = (df["time"] - prev_time).dt.total_seconds() / 60.0
    df["rot"] = np.where((dt_min == 5.0) & prev_tec.notna(),
                         (df["tec"] - prev_tec) / 5.0, np.nan)
    return df.dropna(subset=["rot"])


def compute_roti(df_rot: pd.DataFrame) -> pd.DataFrame:
    """ROTI = std(ROT) trên cửa sổ trượt ROTI_WINDOW_MAPS map 5 phút theo từng ô lưới."""
    n = config.ROTI_WINDOW_MAPS
    df_rot = df_rot.sort_values(["gdlat", "glon", "time"]).copy()
    cols = [df_rot.groupby(["gdlat", "glon"])["rot"].shift(i) for i in range(n)]
    stack = pd.concat(cols, axis=1).to_numpy()
    n_valid = np.sum(~np.isnan(stack), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        std = np.nanstd(stack, axis=1, ddof=1)
    # chỉ tính ROTI khi cửa sổ có >= 2 điểm hợp lệ
    df_rot["roti"] = np.where(n_valid >= 2, std, np.nan)
    df_rot["epb_flag"] = df_rot["roti"] >= config.ROTI_THRESHOLD
    return df_rot


def vn_summary(df_roti: pd.DataFrame) -> pd.DataFrame:
    """Thống kê mỗi 5 phút trong khung VN: số ô EPB-flag, ROTI max, TEC min/max."""
    vn = df_roti[(df_roti.gdlat >= config.LAT_MIN) & (df_roti.gdlat <= config.LAT_MAX) &
                 (df_roti.glon >= config.LON_MIN) & (df_roti.glon <= config.LON_MAX)]
    g = vn.groupby("time")
    out = pd.DataFrame({
        "n_cells": g.size(),
        "n_epb_cells": g["epb_flag"].sum().astype(int),
        "roti_max": g["roti"].max(),
        "tec_mean": g["tec"].mean(),
    }).reset_index()
    out["utc_hour"] = out.time.dt.hour
    return out


def hourly_stats(summary: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp theo giờ UTC — giờ tối VN (12-17 UTC = 19h-24h địa phương) là khung EPB."""
    g = summary.groupby(summary.time.dt.floor("1h"))
    return pd.DataFrame({
        "epb_cells_per_hour": g.n_epb_cells.sum().astype(int),
        "roti_max_per_hour": g.roti_max.max(),
    }).reset_index()


if __name__ == "__main__":
    import argparse
    import importlib

    p = argparse.ArgumentParser(description="Tính ROTI từ 1 file Madrigal TEC hdf5")
    p.add_argument("hdf5", type=Path)
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "out"))
    args = p.parse_args()

    fetch_madrigal = importlib.import_module("src.fetch_madrigal")
    df = fetch_madrigal.load_gridded_tec(args.hdf5)
    print(f"Loaded {len(df):,} rows, {df.time.nunique()} maps")
    df_rot = compute_rot(df)
    df_roti = compute_roti(df_rot)
    summary = vn_summary(df_roti)
    hourly = hourly_stats(summary)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = args.hdf5.stem
    summary.to_parquet(out / f"{stem}_vn_5min.parquet")
    hourly.to_parquet(out / f"{stem}_vn_hourly.parquet")

    print("\n== Theo giờ UTC (khung EPB VN: 12-17 UTC = 19-24h tối địa phương) ==")
    print(hourly.to_string(index=False))
    print(f"\nTổng ô EPB-flag (ROTI>=0.5) cả ngày trong khung VN: {int(summary.n_epb_cells.sum())}")
