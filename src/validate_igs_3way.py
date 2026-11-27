"""Đối chứng 3 chiều: IGS 30s ROTI (chuẩn Pi et al., 1997) vs Madrigal 3500 (ROTI-proxy) vs COSMIC-2 scnLv2 (S4 tầng F).

Chạy trên 3 ngày: 2024-05-10, 2024-05-11, 2024-06-04 trong khung tối VN (12-17 UTC).
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

from src import fetch_madrigal, igs_roti, roti, scnlv2  # noqa: E402

DAYS = [
    ("2024-05-10", 2024, 5, 10, 131),
    ("2024-05-11", 2024, 5, 11, 132),
    ("2024-06-04", 2024, 6, 4, 156),
]


def corr_pair(a: pd.Series, b: pd.Series) -> tuple[int, float, float]:
    m = a.notna() & b.notna()
    n = int(m.sum())
    if n < 3:
        return n, float("nan"), float("nan")
    pearson = float(a[m].corr(b[m], method="pearson"))
    spearman = float(a[m].rank().corr(b[m].rank(), method="pearson"))
    return n, pearson, spearman


def main():
    igs_dir = ROOT / "data" / "raw" / "igs"
    tec_dir = ROOT / "data" / "raw" / "tec"
    c2_dir = ROOT / "data" / "raw" / "cosmic2"
    out_dir = ROOT / "data" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    igs_5min_all = []
    m3500_all = []
    c2_all = []

    print("=== 1. Tính ROTI 30s (Pi et al., 1997) cho 4 trạm IGS (12-17 UTC) ===")
    for ds_str, y, m, d, doy in DAYS:
        # IGS 4 stations
        for fp in sorted(igs_dir.glob(f"*_R_{y}{doy:03d}0000_01D_30S_MO.crx.gz")):
            meta, df_ph = igs_roti.parse_rinex3_gps_phase(fp)
            df_r = igs_roti.compute_igs_roti(df_ph)
            night = df_r[(df_r.utc_hour >= 12) & (df_r.utc_hour <= 17) & df_r.roti.notna()]
            g5 = night.groupby("time_5min").agg(
                igs_roti_mean=("roti", "mean"),
                igs_roti_p95=("roti", lambda s: s.quantile(0.95)),
                igs_roti_max=("roti", "max"),
            ).reset_index()
            g5["station"] = meta["station"]
            g5["lat"] = meta["lat"]
            g5["lon"] = meta["lon"]
            g5["date"] = ds_str
            igs_5min_all.append(g5)
            print(
                f"  {ds_str} {meta['station']} ({meta['lat']:.2f}N, {meta['lon']:.2f}E) "
                f"[{meta['l1_obs']}/{meta['l2_obs']}]: 5min_bins={len(g5)}, "
                f"ROTI p95={night.roti.quantile(0.95):.3f}, max={night.roti.max():.3f}, "
                f">=0.5: {int((night.roti >= 0.5).sum())}"
            )

        # Madrigal 3500
        p3500 = tec_dir / f"gps{y % 100:02d}{m:02d}{d:02d}.hdf5"
        df0 = roti.compute_roti(roti.compute_rot(fetch_madrigal.load_gridded_tec(p3500)))
        df0["utc_hour"] = df0["time"].dt.hour
        df0_n = df0[(df0.utc_hour >= 12) & (df0.utc_hour <= 17) &
                    df0.gdlat.between(0, 28) & df0.glon.between(95, 122)].copy()
        df0_n["time_5min"] = df0_n["time"].dt.floor("5min")
        df0_n["date"] = ds_str
        m3500_all.append(df0_n)

        # COSMIC-2 scnLv2 (F >= 200 km)
        tar_c2 = c2_dir / f"scnLv2_rapid_{y}_{doy:03d}.tar.gz"
        df_c2, _ = scnlv2.parse_scnlv2_tar(tar_c2, lat_min=0.0, lat_max=28.0, lon_min=95.0, lon_max=122.0)
        c2_n = df_c2[(df_c2.utc_hour >= 12) & (df_c2.utc_hour <= 17)].copy() if not df_c2.empty else df_c2
        if not c2_n.empty:
            c2_n["date"] = ds_str
            c2_all.append(c2_n)

    df_igs = pd.concat(igs_5min_all, ignore_index=True)
    df_3500 = pd.concat(m3500_all, ignore_index=True)
    df_c2 = pd.concat(c2_all, ignore_index=True)
    df_igs.to_parquet(out_dir / "igs_roti_3days_12_17utc.parquet")

    # --- 2. Đối chứng cùng ô không gian-thời gian (trong bán kính ±2° quanh trạm IGS, cùng 5 phút) ---
    print("\n=== 2. Tương quan IGS 30s ROTI vs Madrigal 3500 ROTI-proxy (bán kính ±2° quanh trạm, bước 5 phút) ===")
    matched_rows = []
    for st, grp in df_igs.groupby("station"):
        slat, slon = grp["lat"].iloc[0], grp["lon"].iloc[0]
        near = df_3500[
            (df_3500.gdlat.between(slat - 2.0, slat + 2.0)) &
            (df_3500.glon.between(slon - 2.0, slon + 2.0))
        ]
        m_st = near.groupby("time_5min").agg(m3500_roti_mean=("roti", "mean"), m3500_roti_max=("roti", "max")).reset_index()
        merged = pd.merge(grp, m_st, on="time_5min", how="inner")
        matched_rows.append(merged)
        n, r, rho = corr_pair(merged["igs_roti_p95"], merged["m3500_roti_max"])
        print(f"  {st} ({slat:.1f}N, {slon:.1f}E): N={n}, Pearson={r:.3f}, Spearman={rho:.3f}")

    all_m = pd.concat(matched_rows, ignore_index=True)
    n, r, rho = corr_pair(all_m["igs_roti_p95"], all_m["m3500_roti_max"])
    print(f"  -> TỔNG 4 TRẠM (5-min): N={n}, Pearson={r:.3f}, Spearman={rho:.3f}")

    # --- 3. Tổng hợp theo giờ (18 giờ tối = 3 ngày x 6 giờ) giữa 3 nguồn ---
    print("\n=== 3. Tương quan 3 chiều theo giờ (12-17 UTC, 18 giờ tối, khung 0-28N 95-122E) ===")
    h_igs = df_igs.assign(h=df_igs.time_5min.dt.floor("1h")).groupby("h")["igs_roti_p95"].max()
    h_3500 = df_3500.assign(h=df_3500.time.dt.floor("1h")).groupby("h")["roti"].quantile(0.95)
    h_c2 = df_c2.assign(h=df_c2.time.dt.floor("1h")).groupby("h")["s4_L1"].quantile(0.95)
    hdf = pd.concat([h_igs.rename("igs_30s_roti"), h_3500.rename("m3500_proxy_p95"), h_c2.rename("scnlv2_s4_p95")], axis=1)
    for a, b in [
        ("igs_30s_roti", "m3500_proxy_p95"),
        ("igs_30s_roti", "scnlv2_s4_p95"),
        ("m3500_proxy_p95", "scnlv2_s4_p95"),
    ]:
        nh, rh, sh = corr_pair(hdf[a], hdf[b])
        print(f"  {a} vs {b}: N={nh}, Pearson={rh:.3f}, Spearman={sh:.3f}")


if __name__ == "__main__":
    main()
