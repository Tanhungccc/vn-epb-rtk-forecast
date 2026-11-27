"""Batch Ingestion có checkpoint (Tải – Cắt – Nối – Xóa) cho giai đoạn Dataset & Mô hình.

Nguồn tinh gọn theo ADR-011 (~35–60 MB thô/ngày):
  1. ESA GIM IONEX (~1 MB/ngày, zero-auth) -> bản đồ TEC 24h & độ chênh EIA Bắc–Nam
  2. NASA OMNI2 hourly (~3.5 MB/năm, zero-auth) -> F10.7, Bz, v_sw, Ey, Kp, Dst, sin/cos(DOY)
  3. IGS 30s 4 trạm BKG (HKSL, CUSV, PIMO, NTUS, ~15 MB/ngày) -> ROTI chuẩn (Pi et al., 1997), sleep 0.5s
  4. COSMIC-2 scnLv2_rapid (~6–48 MB/ngày) -> Ground Truth S4 tầng F (occheight >= 200 km)
     kèm nhãn tách riêng 3 miền VN (Bắc 18–24N, Trung 12–18N, Nam 6–12N).

Lưu khung giờ 06–17 UTC (13h–24h LT) để train_baseline.py trượt cửa sổ lead_hours = 1..6
trong cùng ngày (groupby date) dự báo cho khung tối 12–17 UTC (19h–24h LT).
"""
import argparse
import json
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import igs_roti, precursors, scnlv2  # noqa: E402

IGS_STATIONS = ("HKSL00HKG", "CUSV00THA", "PIMO00PHL", "NTUS00SGP")
BKG_OBS_BASE = "https://igs.bkg.bund.de/root_ftp/IGS/obs"

VN_BANDS = {
    "north": (18.0, 24.0, True),    # 18 <= lat <= 24
    "central": (12.0, 18.0, False), # 12 <= lat < 18
    "south": (6.0, 12.0, False),    # 6 <= lat < 12
}


def fetch_igs_day(d: date, out_dir: Path) -> list[Path]:
    """Tải 4 trạm IGS 30s (.crx.gz) của ngày d từ BKG với retry + sleep 0.5s chống nghẽn."""
    out_dir.mkdir(parents=True, exist_ok=True)
    doy = d.timetuple().tm_yday
    url_dir = f"{BKG_OBS_BASE}/{d.year}/{doy:03d}/"
    r = requests.get(url_dir, timeout=30)
    r.raise_for_status()
    html = r.text
    paths = []
    for st in IGS_STATIONS:
        m = re.findall(rf'href="({st}[^"]+30S_MO\.crx\.gz)"', html)
        if not m:
            continue
        fname = m[0]
        dest = out_dir / fname
        if dest.exists() and dest.stat().st_size > 500_000:
            paths.append(dest)
            continue
        time.sleep(0.5)
        for att in range(3):
            try:
                fr = requests.get(url_dir + fname, timeout=90)
                fr.raise_for_status()
                dest.write_bytes(fr.content)
                paths.append(dest)
                break
            except Exception:
                time.sleep(2)
    return paths


def _band_stats(df_c2: pd.DataFrame, lat_lo: float, lat_hi: float, inc_hi: bool, s4_thresh: float = 0.30) -> dict:
    if df_c2.empty:
        return {"pts": 0, "s4_p95": np.nan, "s4_max": np.nan, "epb_flag": np.nan}
    m = (df_c2["lat"] >= lat_lo) & (df_c2["lat"] <= lat_hi if inc_hi else df_c2["lat"] < lat_hi)
    sub = df_c2[m]
    if sub.empty:
        return {"pts": 0, "s4_p95": np.nan, "s4_max": np.nan, "epb_flag": np.nan}
    s4_max = float(sub["s4_L1"].max())
    s4_p95 = float(sub["s4_L1"].quantile(0.95))
    return {
        "pts": int(len(sub)),
        "s4_p95": s4_p95,
        "s4_max": s4_max,
        "epb_flag": int(s4_max >= s4_thresh) if np.isfinite(s4_max) else np.nan,
    }


def process_one_day(
    d: date,
    df_omni: pd.DataFrame,
    raw_root: Path,
    keep_raw: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Xử lý trọn vẹn 1 ngày -> (hourly_features_df [06-17 UTC], c2_points_df [12-17 UTC])."""
    doy = d.timetuple().tm_yday
    ds_str = d.isoformat()

    # 1. ESA IONEX -> EIA theo từng giờ (nội suy giờ lẻ nếu là file 2h)
    inx_path = precursors.fetch_esa_ionex(d, raw_root / "ionex")
    df_inx = precursors.parse_ionex_eia_gradient(inx_path)
    df_inx_h = df_inx.groupby("utc_hour")[["tec_eia_north", "tec_trough_south", "eia_diff_ns", "eia_ratio_ns"]].mean()
    df_inx_h = df_inx_h.reindex(range(0, 24)).interpolate(method="linear", limit_direction="both")

    # 2. OMNI2 -> hourly 06-17 UTC
    om_day = df_omni[df_omni["time"].dt.date == d].set_index("hour")

    # 3. IGS 30s -> hourly ROTI per station (06-17 UTC)
    igs_paths = fetch_igs_day(d, raw_root / "igs")
    igs_rows = []
    for fp in igs_paths:
        try:
            meta, df_ph = igs_roti.parse_rinex3_gps_phase(fp)
            df_r = igs_roti.compute_igs_roti(df_ph)
            sub = df_r[(df_r.utc_hour >= 6) & (df_r.utc_hour <= 17) & df_r.roti.notna()]
            if sub.empty:
                continue
            g = sub.groupby("utc_hour")["roti"].agg(
                p95=lambda s: float(s.quantile(0.95)),
                epb_frac=lambda s: float((s >= 0.5).mean()),
            )
            for hr, r_row in g.iterrows():
                igs_rows.append({
                    "station": meta["station"],
                    "utc_hour": int(hr),
                    "roti_p95": float(r_row["p95"]),
                    "epb_frac": float(r_row["epb_frac"]),
                })
        except Exception as e:  # noqa: BLE001
            print(f"  warn IGS {fp.name}: {e}", flush=True)
        finally:
            if not keep_raw and fp.exists():
                fp.unlink()

    df_igs_h = pd.DataFrame(igs_rows)
    igs_piv = {}
    if not df_igs_h.empty:
        for st in ("HKSL", "CUSV", "PIMO", "NTUS"):
            igs_piv[st] = df_igs_h[df_igs_h.station == st].set_index("utc_hour")

    # 4. COSMIC-2 scnLv2_rapid (F-layer >= 200 km, khung rộng 0-28N 95-122E)
    c2_tar = scnlv2.fetch_scnlv2(d.year, doy, raw_root / "cosmic2")
    df_c2, _ = scnlv2.parse_scnlv2_tar(
        c2_tar, lat_min=0.0, lat_max=28.0, lon_min=95.0, lon_max=122.0, min_occheight_km=200.0
    )
    if not keep_raw and c2_tar.exists():
        c2_tar.unlink()

    c2_eve = df_c2[(df_c2.utc_hour >= 12) & (df_c2.utc_hour <= 17)].copy() if not df_c2.empty else pd.DataFrame()
    if not c2_eve.empty:
        c2_eve["date"] = ds_str

    # 5. Ghép bảng đặc trưng theo từng giờ 06-17 UTC (hỗ trợ lead 1..6h cho mục tiêu 12-17 UTC)
    hourly_rows = []
    for hr in range(6, 18):
        om_r = om_day.loc[hr] if hr in om_day.index else None
        inx_r = df_inx_h.loc[hr] if hr in df_inx_h.index else None

        c2_h = df_c2[df_c2.utc_hour == hr] if not df_c2.empty else pd.DataFrame()
        c2_vn = c2_h[c2_h.lat.between(6, 24) & c2_h.lon.between(101, 116)] if not c2_h.empty else pd.DataFrame()

        row = {
            "date": ds_str,
            "year": d.year,
            "month": d.month,
            "doy": doy,
            "utc_hour": hr,
            "lt_hour": (hr + 7) % 24,
            "sin_doy": float(om_r["sin_doy"]) if om_r is not None else np.nan,
            "cos_doy": float(om_r["cos_doy"]) if om_r is not None else np.nan,
            "f107": float(om_r["f107"]) if om_r is not None else np.nan,
            "bz_gsm": float(om_r["bz_gsm"]) if om_r is not None else np.nan,
            "v_sw": float(om_r["v_sw"]) if om_r is not None else np.nan,
            "ey_mvm": float(om_r["ey_mvm"]) if om_r is not None else np.nan,
            "kp": float(om_r["kp"]) if om_r is not None else np.nan,
            "dst": float(om_r["dst"]) if om_r is not None else np.nan,
            "eia_north": float(inx_r["tec_eia_north"]) if inx_r is not None else np.nan,
            "eia_south": float(inx_r["tec_trough_south"]) if inx_r is not None else np.nan,
            "eia_diff": float(inx_r["eia_diff_ns"]) if inx_r is not None else np.nan,
            "eia_ratio": float(inx_r["eia_ratio_ns"]) if inx_r is not None else np.nan,
            # Nhãn chung khu vực (0-28N, 95-122E) & toàn VN (6-24N, 101-116E)
            "c2_reg_pts": int(len(c2_h)),
            "c2_reg_s4_p95": float(c2_h["s4_L1"].quantile(0.95)) if not c2_h.empty else np.nan,
            "c2_reg_s4_max": float(c2_h["s4_L1"].max()) if not c2_h.empty else np.nan,
            "c2_vn_pts": int(len(c2_vn)),
            "c2_vn_s4_p95": float(c2_vn["s4_L1"].quantile(0.95)) if not c2_vn.empty else np.nan,
            "c2_vn_s4_max": float(c2_vn["s4_L1"].max()) if not c2_vn.empty else np.nan,
        }
        # Nhãn tách riêng 3 miền VN (101-116E)
        for b_name, (lo, hi, inc) in VN_BANDS.items():
            bs = _band_stats(c2_vn, lo, hi, inc)
            row[f"c2_{b_name}_pts"] = bs["pts"]
            row[f"c2_{b_name}_s4_p95"] = bs["s4_p95"]
            row[f"c2_{b_name}_s4_max"] = bs["s4_max"]
            row[f"c2_{b_name}_epb"] = bs["epb_flag"]

        for st in ("HKSL", "CUSV", "PIMO", "NTUS"):
            s_df = igs_piv.get(st)
            row[f"igs_{st.lower()}_roti"] = float(s_df.loc[hr, "roti_p95"]) if (s_df is not None and hr in s_df.index) else np.nan
        hourly_rows.append(row)

    return pd.DataFrame(hourly_rows), c2_eve


def daterange(start: date, end: date) -> list[date]:
    out = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur += timedelta(days=1)
    return out


def main():
    ap = argparse.ArgumentParser(description="Batch ingestion có checkpoint cho 2024–2026")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--keep-raw", action="store_true", help="Giữ lại file thô .crx.gz và .tar.gz")
    args = ap.parse_args()

    s_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    e_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    raw_root = ROOT / "data" / "raw"
    ds_dir = ROOT / "data" / "dataset"
    ds_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ds_dir / "checkpoint.json"
    feat_pq = ds_dir / "hourly_features.parquet"
    c2_pq = ds_dir / "scnlv2_points.parquet"

    done_days = set(json.loads(ckpt_path.read_text())) if ckpt_path.exists() else set()

    years = sorted({d.year for d in daterange(s_date, e_date)})
    omni_frames = [precursors.load_omni2_hourly(precursors.fetch_omni2_year(y, raw_root / "omni")) for y in years]
    df_omni = pd.concat(omni_frames, ignore_index=True)

    all_feat = [pd.read_parquet(feat_pq)] if feat_pq.exists() else []
    all_c2 = [pd.read_parquet(c2_pq)] if c2_pq.exists() else []

    for d in daterange(s_date, e_date):
        ds = d.isoformat()
        if ds in done_days:
            print(f"SKIP {ds} (checkpoint)", flush=True)
            continue
        print(f"Processing {ds}...", flush=True)
        try:
            df_h, df_c2 = process_one_day(d, df_omni, raw_root, keep_raw=args.keep_raw)
            all_feat.append(df_h)
            if not df_c2.empty:
                all_c2.append(df_c2)
            pd.concat(all_feat, ignore_index=True).drop_duplicates(["date", "utc_hour"]).to_parquet(feat_pq)
            if all_c2:
                pd.concat(all_c2, ignore_index=True).to_parquet(c2_pq)
            done_days.add(ds)
            ckpt_path.write_text(json.dumps(sorted(done_days), indent=2))
            print(f"  OK {ds}: {len(df_h)}h rows, c2_pts={len(df_c2)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {ds}: {e}", flush=True)

    print(f"Done. Total days in checkpoint: {len(done_days)} -> {feat_pq}")


if __name__ == "__main__":
    main()
