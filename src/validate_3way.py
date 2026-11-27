"""Kiểm chứng 3 chiều: COSMIC-2 scnLv2 (S4 tầng F) vs Madrigal 3507 (ROTI 30s) vs 3500 (ROTI-proxy).

Chạy trên 4 ngày: 2024-05-10, 2024-05-11, 2024-06-04, 2026-11-20.
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config  # noqa: E402
from src import fetch_madrigal, roti, roti_3507, scnlv2  # noqa: E402

DATES_4 = ["2024-05-10", "2024-05-11", "2024-06-04", "2026-11-20"]
DATES_3 = ["2024-05-10", "2024-05-11", "2024-06-04"]


def band_coverage(df_grid: pd.DataFrame, lon_min: float, lon_max: float, lat_lo: float, lat_hi: float, inclusive_hi: bool) -> tuple[int, int, float]:
    """Tính số ô non-NaN / tổng ô-slot 1°x1°x5min trong khung giờ 12-17 UTC (72 maps)."""
    lat_mask = (df_grid["lat_bin"] >= lat_lo) & (
        df_grid["lat_bin"] <= lat_hi if inclusive_hi else df_grid["lat_bin"] < lat_hi
    )
    sub = df_grid[
        lat_mask & (df_grid["lon_bin"] >= lon_min) & (df_grid["lon_bin"] <= lon_max) &
        (df_grid["utc_hour"] >= 12) & (df_grid["utc_hour"] <= 17)
    ]
    n_lats = int(lat_hi - lat_lo + (1 if inclusive_hi else 0))
    n_lons = int(lon_max - lon_min + 1)
    total_slots = n_lats * n_lons * 72
    filled = sub.drop_duplicates(subset=["time_5min", "lat_bin", "lon_bin"]).shape[0]
    return filled, total_slots, 100.0 * filled / total_slots


def corr_stats(a: pd.Series, b: pd.Series) -> tuple[int, float, float]:
    m = a.notna() & b.notna()
    n = int(m.sum())
    if n < 3:
        return n, float("nan"), float("nan")
    return n, float(a[m].corr(b[m], method="pearson")), float(a[m].corr(b[m], method="spearman"))


def main():
    raw_c2 = ROOT / "data" / "raw" / "cosmic2"
    raw_tec = ROOT / "data" / "raw" / "tec"
    out_dir = ROOT / "data" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. COSMIC-2 scnLv2_rapid (4 ngày) ---
    print("================ 1. COSMIC-2 scnLv2_rapid (F-layer >= 200 km) ================")
    c2_night_frames = []
    for ds in DATES_4:
        dt = datetime.strptime(ds, "%Y-%m-%d")
        doy = dt.timetuple().tm_yday
        tar = scnlv2.fetch_scnlv2(dt.year, doy, raw_c2)
        df_c2, st = scnlv2.parse_scnlv2_tar(tar)
        night = df_c2[(df_c2.utc_hour >= 12) & (df_c2.utc_hour <= 17)] if not df_c2.empty else df_c2
        s4_max = float(night.s4_L1.max()) if not night.empty else float("nan")
        s4_p95 = float(night.s4_L1.quantile(0.95)) if not night.empty else float("nan")
        n_ge03 = int((night.s4_L1 >= 0.3).sum()) if not night.empty else 0
        print(
            f"{ds} (DOY {doy:03d}): {st['tar_mb']:.2f} MB | global={st['total_files']} files "
            f"(bitstatus_ok={st['bitstatus_ok_files']}, LEOs={st['leo_ids']}) | "
            f"VN_raw={st['vn_raw_files']}f/{st['vn_raw_pts']}p -> F>=200km={st['vn_f_files']}f/{st['vn_f_pts']}p "
            f"(RFI_drop={st['vn_rfi_dropped']}) | 12-17UTC={st['vn_night_files']}f/{st['vn_night_pts']}p | "
            f"s4_L1 max={s4_max:.3f}, p95={s4_p95:.3f}, >=0.3: {n_ge03}"
        )
        if not night.empty:
            night = night.copy()
            night["date"] = ds
            c2_night_frames.append(night)

    df_c2_all = pd.concat(c2_night_frames, ignore_index=True) if c2_night_frames else pd.DataFrame()
    if not df_c2_all.empty:
        df_c2_all["time_5min"] = df_c2_all["time"].dt.floor("5min")
        df_c2_all["lat_bin"] = np.round(df_c2_all["lat"]).astype(int)
        df_c2_all["lon_bin"] = np.round(df_c2_all["lon"]).astype(int)

    # --- 2. Madrigal 3507 vs 3500: độ phủ 2 dải vĩ độ & dung lượng parquet ---
    print("\n================ 2. Madrigal 3507 & 3500 Coverage (12-17 UTC) ================")
    frames_3507 = []
    frames_3500 = []
    for ds in DATES_3:
        dt = datetime.strptime(ds, "%Y-%m-%d")
        doy = dt.timetuple().tm_yday
        p3500 = raw_tec / f"gps{dt.year % 100:02d}{dt.month:02d}{dt.day:02d}.hdf5"
        p3507 = raw_tec / f"roti_{doy:03d}_{dt.year}.h5"

        df0 = fetch_madrigal.load_gridded_tec(p3500)
        df0_roti = roti.compute_roti(roti.compute_rot(df0))
        df0_roti["utc_hour"] = df0_roti["time"].dt.hour
        df0_roti["time_5min"] = df0_roti["time"]
        df0_roti["lat_bin"] = df0_roti["gdlat"].astype(int)
        df0_roti["lon_bin"] = df0_roti["glon"].astype(int)
        df0_roti["date"] = ds
        frames_3500.append(df0_roti)

        df7 = roti_3507.load_roti_3507_region(p3507, lat_min=6.0, lat_max=24.0, lon_min=99.0, lon_max=122.0)
        pq_ext = out_dir / f"roti3507_{ds}_6_24N_99_122E.parquet"
        pq_vn = out_dir / f"roti3507_{ds}_6_24N_101_116E.parquet"
        df7.to_parquet(pq_ext)
        df7[(df7.glon >= 101.0) & (df7.glon <= 116.0)].to_parquet(pq_vn)
        print(
            f"{ds}: 3507 cropped parquet -> 101-116E: {pq_vn.stat().st_size/1e6:.2f} MB ({len(df7[(df7.glon>=101)&(df7.glon<=116)]):,} rows) | "
            f"99-122E: {pq_ext.stat().st_size/1e6:.2f} MB ({len(df7):,} rows)"
        )
        df7["date"] = ds
        frames_3507.append(df7)

        for label, src_df in [("3500 (TEC 5m)", df0_roti), ("3507 (ROTI 30s)", df7)]:
            for lon_tag, lo, hi in [("101-116E", 101, 116), ("99-122E", 99, 122)]:
                fn, tn, pn = band_coverage(src_df, lo, hi, 18, 24, True)
                fs, ts, ps = band_coverage(src_df, lo, hi, 6, 18, False)
                print(
                    f"  {label:15s} [{lon_tag}] | Bac (18-24N): {pn:5.2f}% ({fn}/{tn}) | "
                    f"Trung-Nam (6-18N): {ps:5.2f}% ({fs}/{ts})"
                )

    # --- 3. Đối chứng 3 chiều (12-17 UTC trên 3 ngày 2024) ---
    print("\n================ 3. 3-Way Correlation (12-17 UTC, 2024-05-10/11 & 06-04) ================")
    all_3500 = pd.concat(frames_3500, ignore_index=True)
    all_3500_n = all_3500[(all_3500.utc_hour >= 12) & (all_3500.utc_hour <= 17) &
                          (all_3500.lat_bin >= 6) & (all_3500.lat_bin <= 24) &
                          (all_3500.lon_bin >= 99) & (all_3500.lon_bin <= 122)]

    all_3507 = pd.concat(frames_3507, ignore_index=True)
    all_3507_n = all_3507[(all_3507.utc_hour >= 12) & (all_3507.utc_hour <= 17)]

    g3507 = all_3507_n.groupby(["time_5min", "lat_bin", "lon_bin"]).agg(
        roti3507_mean=("roti_tecu_min", "mean"),
        roti3507_max=("roti_tecu_min", "max"),
    ).reset_index()

    g3500 = all_3500_n[["time_5min", "lat_bin", "lon_bin", "roti"]].rename(columns={"roti": "roti3500"})

    # 3507 vs 3500 cell-level
    m_rot = pd.merge(g3507, g3500, on=["time_5min", "lat_bin", "lon_bin"], how="inner")
    n1, r1, s1 = corr_stats(m_rot["roti3507_mean"], m_rot["roti3500"])
    n1m, r1m, s1m = corr_stats(m_rot["roti3507_max"], m_rot["roti3500"])
    print(f"Cell-level (1deg x 5min) 3507_mean vs 3500: N={n1:,}, Pearson={r1:.3f}, Spearman={s1:.3f}")
    print(f"Cell-level (1deg x 5min) 3507_max  vs 3500: N={n1m:,}, Pearson={r1m:.3f}, Spearman={s1m:.3f}")

    # COSMIC-2 scnLv2 s4_L1 vs 3507 & 3500 (binned 1deg x 5min, và 1deg x 15min)
    c2_2024 = df_c2_all[df_c2_all["date"].isin(DATES_3)].copy()
    gc2 = c2_2024.groupby(["time_5min", "lat_bin", "lon_bin"]).agg(
        s4_max=("s4_L1", "max"),
        s4_mean=("s4_L1", "mean"),
    ).reset_index()

    m_c2_7 = pd.merge(gc2, g3507, on=["time_5min", "lat_bin", "lon_bin"], how="inner")
    m_c2_0 = pd.merge(gc2, g3500, on=["time_5min", "lat_bin", "lon_bin"], how="inner")
    n2, r2, s2 = corr_stats(m_c2_7["s4_max"], m_c2_7["roti3507_max"])
    n3, r3, s3 = corr_stats(m_c2_0["s4_max"], m_c2_0["roti3500"])
    print(f"Cell-level (1deg x 5min) S4_max vs 3507_max: N={n2:,}, Pearson={r2:.3f}, Spearman={s2:.3f}")
    print(f"Cell-level (1deg x 5min) S4_max vs 3500    : N={n3:,}, Pearson={r3:.3f}, Spearman={s3:.3f}")

    # Hourly VN-aggregate (101-116E, 6-24N)
    for name, df_sub in [("3507", all_3507_n[all_3507_n.glon.between(101, 116)]),
                         ("3500", all_3500_n[all_3500_n.lon_bin.between(101, 116)]),
                         ("c2", c2_2024)]:
        df_sub["hour_utc"] = df_sub["time"].dt.floor("1h")
    h7 = all_3507_n[all_3507_n.glon.between(101, 116)].assign(h=lambda d: d.time.dt.floor("1h")).groupby("h")["roti_tecu_min"].quantile(0.95)
    h0 = all_3500_n[all_3500_n.lon_bin.between(101, 116)].assign(h=lambda d: d.time.dt.floor("1h")).groupby("h")["roti"].quantile(0.95)
    hc = c2_2024.assign(h=lambda d: d.time.dt.floor("1h")).groupby("h")["s4_L1"].quantile(0.95)
    hdf = pd.concat([h7.rename("roti3507_p95"), h0.rename("roti3500_p95"), hc.rename("s4_p95")], axis=1)
    print(f"Hourly VN-box (p95, 18 night hours):")
    for a, b in [("roti3507_p95", "roti3500_p95"), ("s4_p95", "roti3507_p95"), ("s4_p95", "roti3500_p95")]:
        nh, rh, sh = corr_stats(hdf[a], hdf[b])
        print(f"  {a} vs {b}: N={nh}, Pearson={rh:.3f}, Spearman={sh:.3f}")


if __name__ == "__main__":
    main()
