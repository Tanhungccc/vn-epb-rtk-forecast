"""Luồng suy luận Real-Time (Live Forecast) cho Web App dự báo EPB Việt Nam.

Đặc điểm thực tế:
  1. Logic theo giờ Việt Nam (UTC+7):
     - Ban ngày (06:00–18:59 LT): Chế độ "Dự báo Sớm cho Khung Tối Nay (19:00–24:00 LT)".
     - Ban đêm (19:00–23:59 LT): Chế độ "Nowcast Trực Tiếp (+1h .. +6h)".
  2. Chuỗi trễ thực tế t-1h .. t-6h từ NOAA SWPC (Bz, v_sw, Ey, Kp, F10.7, Dst).
  3. Hourly IGS 30s từ BKG NRT (2–3 giờ gần nhất) cho 4 trạm HKSL, CUSV, PIMO, NTUS;
     trạm nào chưa kịp cập nhật thì fallback NaN riêng cho trạm đó.
  4. Tính chỉ số độ tin cậy dữ liệu (data_confidence 0–100%) và tự động lưu nhật ký
     dự báo vào data/forecast_log.parquet (chống ghi trùng trong cùng 1 giờ UTC).
"""
import math
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import igs_roti, precursors  # noqa: E402

VN_TZ = timezone(timedelta(hours=7))

SWPC_URLS = {
    "mag": "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json",
    "wind": "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json",
    "kp": "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json",
    "f107": "https://services.swpc.noaa.gov/json/f107_cm_flux.json",
    "dst": "https://services.swpc.noaa.gov/products/kyoto-dst.json",
}

BKG_NRT_BASE = "https://igs.bkg.bund.de/root_ftp/IGS/nrt"
BKG_OBS_BASE = "https://igs.bkg.bund.de/root_ftp/IGS/obs"
IGS_STATIONS = ("HKSL", "CUSV", "PIMO", "NTUS")
FORECAST_LOG_PATH = ROOT / "data" / "forecast_log.parquet"


def _prep_swpc_df(raw_json: list, time_col: str = "time_tag", prefer_active: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(raw_json)
    if df.empty or time_col not in df.columns:
        return pd.DataFrame()
    if prefer_active and "active" in df.columns:
        act = df[df["active"] == True]  # noqa: E712
        if not act.empty:
            df = act.copy()
    df["_t"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    return df.dropna(subset=["_t"]).sort_values("_t").reset_index(drop=True)


def _hourly_bucket_stats(df: pd.DataFrame, col: str, now_utc: datetime, max_lag: int = 6) -> dict[int, float]:
    """Chia giỏ 1 giờ lùi từ now_utc: lag 0 = [now-1h, now], lag k = [now-(k+1)h, now-kh]."""
    out: dict[int, float] = {}
    if df.empty or col not in df.columns:
        return out
    s = pd.to_numeric(df[col], errors="coerce")
    valid = df[s.notna()].copy()
    if valid.empty:
        return out
    valid["_val"] = s[s.notna()]
    for lag in range(0, max_lag + 1):
        t_hi = now_utc - timedelta(hours=lag)
        t_lo = t_hi - timedelta(hours=1)
        sub = valid[(valid["_t"] > t_lo) & (valid["_t"] <= t_hi)]
        if not sub.empty:
            out[lag] = float(sub["_val"].mean())
    return out


def fetch_live_swpc(now_utc: datetime) -> tuple[dict, dict[int, dict], dict]:
    """Kéo dữ liệu realtime + chuỗi trễ 6h từ NOAA SWPC.

    Trả về: (latest_vals_dict, swpc_lags_by_hour, freshness_dict).
    """
    vals = {
        "bz_gsm": float("nan"), "v_sw": float("nan"), "ey_mvm": float("nan"),
        "kp": float("nan"), "f107": float("nan"), "dst": float("nan"),
        "bz_min_6h": float("nan"), "ey_max_6h": float("nan"), "dst_min_6h": float("nan"),
    }
    lags: dict[int, dict[str, float]] = {k: {} for k in range(0, 7)}
    times: list[datetime] = []
    errors: list[str] = []

    df_mag = pd.DataFrame()
    df_wind = pd.DataFrame()

    try:
        df_mag = _prep_swpc_df(requests.get(SWPC_URLS["mag"], timeout=15).json())
        if not df_mag.empty and "bz_gsm" in df_mag.columns:
            bz_num = pd.to_numeric(df_mag["bz_gsm"], errors="coerce")
            v_mag = df_mag[bz_num.notna()]
            if not v_mag.empty:
                vals["bz_gsm"] = float(bz_num[bz_num.notna()].iloc[-1])
                times.append(v_mag["_t"].iloc[-1].to_pydatetime())
                w6 = v_mag[v_mag["_t"] >= now_utc - timedelta(hours=6)]
                if not w6.empty:
                    vals["bz_min_6h"] = float(pd.to_numeric(w6["bz_gsm"], errors="coerce").min())
            for k, v in _hourly_bucket_stats(df_mag, "bz_gsm", now_utc).items():
                lags[k]["bz_gsm"] = v
    except Exception as e:  # noqa: BLE001
        errors.append(f"mag:{e}")

    try:
        df_wind = _prep_swpc_df(requests.get(SWPC_URLS["wind"], timeout=15).json())
        if not df_wind.empty and "proton_speed" in df_wind.columns:
            v_num = pd.to_numeric(df_wind["proton_speed"], errors="coerce")
            v_w = df_wind[v_num.notna()]
            if not v_w.empty:
                vals["v_sw"] = float(v_num[v_num.notna()].iloc[-1])
                times.append(v_w["_t"].iloc[-1].to_pydatetime())
            for k, v in _hourly_bucket_stats(df_wind, "proton_speed", now_utc).items():
                lags[k]["v_sw"] = v
    except Exception as e:  # noqa: BLE001
        errors.append(f"wind:{e}")

    # Tính Ey = -v_sw * Bz * 1e-3 theo từng phút khớp mag & wind để giữ nguyên biến thiên 6h
    if not df_mag.empty and not df_wind.empty and "bz_gsm" in df_mag.columns and "proton_speed" in df_wind.columns:
        m_sub = df_mag[["_t", "bz_gsm"]].copy()
        w_sub = df_wind[["_t", "proton_speed"]].copy()
        m_sub["bz_gsm"] = pd.to_numeric(m_sub["bz_gsm"], errors="coerce")
        w_sub["proton_speed"] = pd.to_numeric(w_sub["proton_speed"], errors="coerce")
        m_sub = m_sub.dropna().sort_values("_t")
        w_sub = w_sub.dropna().sort_values("_t")
        if not m_sub.empty and not w_sub.empty:
            merged = pd.merge_asof(m_sub, w_sub, on="_t", direction="nearest", tolerance=pd.Timedelta("5min")).dropna()
            if not merged.empty:
                merged["ey_mvm"] = -merged["proton_speed"] * merged["bz_gsm"] * 1e-3
                vals["ey_mvm"] = float(merged["ey_mvm"].iloc[-1])
                w6_ey = merged[merged["_t"] >= now_utc - timedelta(hours=6)]
                if not w6_ey.empty:
                    vals["ey_max_6h"] = float(w6_ey["ey_mvm"].max())
                for k, v in _hourly_bucket_stats(merged, "ey_mvm", now_utc).items():
                    lags[k]["ey_mvm"] = v
    if not np.isfinite(vals["ey_mvm"]) and np.isfinite(vals["bz_gsm"]) and np.isfinite(vals["v_sw"]):
        vals["ey_mvm"] = -vals["v_sw"] * vals["bz_gsm"] * 1e-3

    try:
        df_kp = _prep_swpc_df(requests.get(SWPC_URLS["kp"], timeout=15).json(), prefer_active=False)
        kp_col = "kp_index" if "kp_index" in df_kp.columns else "estimated_kp"
        if not df_kp.empty and kp_col in df_kp.columns:
            kp_num = pd.to_numeric(df_kp[kp_col], errors="coerce")
            v_kp = df_kp[kp_num.notna()]
            if not v_kp.empty:
                vals["kp"] = float(kp_num[kp_num.notna()].iloc[-1])
                times.append(v_kp["_t"].iloc[-1].to_pydatetime())
            for k, v in _hourly_bucket_stats(df_kp, kp_col, now_utc).items():
                lags[k]["kp"] = v
    except Exception as e:  # noqa: BLE001
        errors.append(f"kp:{e}")

    try:
        df_f = _prep_swpc_df(requests.get(SWPC_URLS["f107"], timeout=15).json(), prefer_active=False)
        if not df_f.empty and "flux" in df_f.columns:
            f_num = pd.to_numeric(df_f["flux"], errors="coerce").dropna()
            if not f_num.empty:
                vals["f107"] = float(f_num.iloc[-1])
                for k in range(0, 7):
                    lags[k]["f107"] = vals["f107"]
    except Exception as e:  # noqa: BLE001
        errors.append(f"f107:{e}")

    try:
        df_dst = _prep_swpc_df(requests.get(SWPC_URLS["dst"], timeout=15).json(), prefer_active=False)
        if not df_dst.empty and "dst" in df_dst.columns:
            d_num = pd.to_numeric(df_dst["dst"], errors="coerce")
            v_dst = df_dst[d_num.notna()]
            if not v_dst.empty:
                vals["dst"] = float(d_num[d_num.notna()].iloc[-1])
                w6_d = v_dst[v_dst["_t"] >= now_utc - timedelta(hours=6)]
                if not w6_d.empty:
                    vals["dst_min_6h"] = float(pd.to_numeric(w6_d["dst"], errors="coerce").min())
            for k, v in _hourly_bucket_stats(df_dst, "dst", now_utc).items():
                lags[k]["dst"] = v
    except Exception as e:  # noqa: BLE001
        errors.append(f"dst:{e}")

    # Điền fallback mượt cho các giờ thiếu trong bảng lag bằng giá trị gần nhất của chính biến đó
    for k in range(0, 7):
        for col in ("bz_gsm", "v_sw", "ey_mvm", "kp", "f107", "dst"):
            if col not in lags[k] or not np.isfinite(lags[k][col]):
                lags[k][col] = vals[col]

    if times:
        latest_t = max(times)
        age_min = max(0, int((now_utc - latest_t).total_seconds() // 60))
        swpc_status = f"Live ({age_min} phút trước, chuỗi 6h)" if age_min < 120 else f"Live ({latest_t.strftime('%m-%d %H:%M UTC')})"
    else:
        swpc_status = f"Lỗi kết nối ({', '.join(errors[:2])})"

    return vals, lags, {"swpc": swpc_status}


def fetch_latest_ionex_or_nan(now_utc: datetime) -> tuple[dict, str, bool]:
    """Thử lấy ESA IONEX hôm nay hoặc hôm qua; nếu chưa có thì trả NaN sạch."""
    raw_inx = ROOT / "data" / "raw" / "ionex"
    for delta in (0, 1, 2):
        d = (now_utc - timedelta(days=delta)).date()
        try:
            p = precursors.fetch_esa_ionex(d, raw_inx)
            df_inx = precursors.parse_ionex_eia_gradient(p)
            aft = df_inx[(df_inx.utc_hour >= 7) & (df_inx.utc_hour <= 10)]
            if not aft.empty:
                return {
                    "eia_north": float(aft["tec_eia_north"].mean()),
                    "eia_south": float(aft["tec_trough_south"].mean()),
                    "eia_diff": float(aft["eia_diff_ns"].mean()),
                    "eia_ratio": float(aft["eia_ratio_ns"].mean()),
                }, f"ESA IONEX ({d.isoformat()})", True
        except Exception:
            continue
    return {
        "eia_north": float("nan"), "eia_south": float("nan"),
        "eia_diff": float("nan"), "eia_ratio": float("nan"),
    }, "Fallback NaN (chưa có IONEX mới)", False


def fetch_hourly_igs_roti(now_utc: datetime) -> tuple[dict[str, float], str, int]:
    """Kéo RINEX 30s theo giờ (2–3 giờ gần nhất) từ BKG cho 4 trạm HKSL/CUSV/PIMO/NTUS.

    Trả về: (igs_roti_dict, status_str, n_live_stations).
    """
    out_dir = ROOT / "data" / "raw" / "igs_live"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lấy 3 giờ gần nhất để tránh khuyết biên cửa sổ ROTI 5 phút / độ trễ đẩy file lên BKG
    base_hr = now_utc.replace(minute=0, second=0, microsecond=0)
    hour_slots = [base_hr - timedelta(hours=h) for h in (0, 1, 2)]

    dir_listings: list[tuple[str, str]] = []
    for dt_hr in hour_slots:
        doy = dt_hr.timetuple().tm_yday
        for base in (BKG_NRT_BASE, BKG_OBS_BASE):
            url_dir = (
                f"{base}/{doy:03d}/{dt_hr.hour:02d}/"
                if base == BKG_NRT_BASE
                else f"{base}/{dt_hr.year}/{doy:03d}/{dt_hr.hour:02d}/"
            )
            try:
                resp = requests.get(url_dir, timeout=10)
                if resp.status_code == 200:
                    dir_listings.append((url_dir, resp.text))
                    break
            except Exception:
                continue

    igs_vals = {f"igs_{st.lower()}_roti": float("nan") for st in IGS_STATIONS}
    live_stations: list[str] = []

    for st in IGS_STATIONS:
        phase_frames = []
        for url_dir, html in dir_listings:
            matches = re.findall(rf'href="({st}[^"]+30S_MO\.crx\.gz)"', html, flags=re.I)
            if not matches:
                continue
            fname = matches[0]
            dest = out_dir / fname
            if not (dest.exists() and dest.stat().st_size > 10_000):
                try:
                    fr = requests.get(url_dir + fname, timeout=20)
                    if fr.status_code == 200 and len(fr.content) > 10_000:
                        dest.write_bytes(fr.content)
                except Exception:
                    continue
            if dest.exists() and dest.stat().st_size > 10_000:
                try:
                    _, df_ph = igs_roti.parse_rinex3_gps_phase(dest)
                    if not df_ph.empty:
                        phase_frames.append(df_ph)
                except Exception:
                    continue

        if phase_frames:
            df_all = pd.concat(phase_frames, ignore_index=True).drop_duplicates(subset=["time", "sat"])
            df_r = igs_roti.compute_igs_roti(df_all)
            valid_r = df_r[df_r["roti"].notna()]
            if not valid_r.empty:
                igs_vals[f"igs_{st.lower()}_roti"] = float(valid_r["roti"].quantile(0.95))
                live_stations.append(st)

    n_live = len(live_stations)
    if n_live == len(IGS_STATIONS):
        igs_status = f"Live BKG 4/4 trạm ({', '.join(live_stations)})"
    elif n_live > 0:
        missing = [s for s in IGS_STATIONS if s not in live_stations]
        igs_status = f"Live {n_live}/4 ({', '.join(live_stations)}) | NaN: {', '.join(missing)}"
    else:
        igs_status = "Fallback NaN (BKG chưa có file giờ mới)"

    return igs_vals, igs_status, n_live


def compute_data_confidence(swpc_vals: dict, ionex_ok: bool, n_igs_live: int) -> int:
    """Tính độ đầy đủ dữ liệu đầu vào (0–100%).

    - 5 luồng SWPC (bz_gsm, v_sw, kp, dst, f107): 50% (10% mỗi luồng).
    - ESA IONEX: 10%.
    - 4 trạm IGS 30s: 40% (10% mỗi trạm).
    """
    score = 0
    for k in ("bz_gsm", "v_sw", "kp", "dst", "f107"):
        if np.isfinite(swpc_vals.get(k, float("nan"))):
            score += 10
    if ionex_ok:
        score += 10
    score += 10 * min(4, max(0, int(n_igs_live)))
    return int(min(100, score))


def _append_forecast_log(now_utc: datetime, mode_label: str, confidence: int, swpc_vals: dict, fc_df: pd.DataFrame) -> None:
    """Ghi nhật ký dự báo vào data/forecast_log.parquet (chống trùng theo giờ UTC)."""
    if fc_df.empty:
        return
    run_hour = now_utc.replace(minute=0, second=0, microsecond=0).isoformat()
    log_df = fc_df.copy()
    log_df["run_hour_utc"] = run_hour
    log_df["run_time_utc"] = now_utc.isoformat()
    log_df["mode_label"] = mode_label
    log_df["data_confidence"] = int(confidence)
    for k in ("bz_gsm", "v_sw", "ey_mvm", "kp", "dst", "f107"):
        log_df[k] = float(swpc_vals.get(k, float("nan")))

    FORECAST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if FORECAST_LOG_PATH.exists():
        try:
            old_df = pd.read_parquet(FORECAST_LOG_PATH)
            combined = pd.concat([old_df, log_df], ignore_index=True)
        except Exception:
            combined = log_df
    else:
        combined = log_df

    combined = combined.drop_duplicates(
        subset=["run_hour_utc", "target", "lead", "lt_label"], keep="last"
    ).reset_index(drop=True)
    combined.to_parquet(FORECAST_LOG_PATH, index=False)


def _shap_for_row(bundle: dict, X_row: pd.DataFrame) -> pd.Series | None:
    base = bundle.get("base_model")
    cols = bundle.get("feat_cols", [])
    if base is None:
        return None
    if hasattr(base, "get_booster"):
        vals = base.get_booster().predict(xgb.DMatrix(X_row[cols]), pred_contribs=True)[0][:-1]
        return pd.Series(vals, index=cols)
    if hasattr(base, "predict"):
        vals = base.predict(X_row[cols], pred_contrib=True)[0][:-1]
        return pd.Series(vals, index=cols)
    return None


def run_live_forecast(now_vn: datetime | None = None, save_log: bool = True) -> dict:
    """Chạy suy luận live cho toàn VN & 3 miền theo logic ngày/đêm Việt Nam."""
    if now_vn is None:
        now_vn = datetime.now(VN_TZ)
    now_utc = now_vn.astimezone(timezone.utc)
    doy = now_utc.timetuple().tm_yday
    sin_doy = math.sin(2.0 * math.pi * doy / 365.25)
    cos_doy = math.cos(2.0 * math.pi * doy / 365.25)
    cos_2doy = 2.0 * (cos_doy ** 2) - 1.0

    swpc_vals, swpc_lags, freshness = fetch_live_swpc(now_utc)
    ionex_vals, inx_status, ionex_ok = fetch_latest_ionex_or_nan(now_utc)
    igs_vals, igs_status, n_igs_live = fetch_hourly_igs_roti(now_utc)
    freshness["ionex"] = inx_status
    freshness["igs"] = igs_status

    data_confidence = compute_data_confidence(swpc_vals, ionex_ok, n_igs_live)

    # Tích phân Ey dương tích lũy trong 6h gần nhất và tốc độ biến thiên 3h từ chuỗi SWPC
    ey_pos_sum = sum(
        max(0.0, float(swpc_lags[k].get("ey_mvm", 0.0)))
        for k in range(0, 6)
        if np.isfinite(swpc_lags[k].get("ey_mvm", float("nan")))
    )
    d_bz_3h = swpc_lags[0].get("bz_gsm", float("nan")) - swpc_lags[3].get("bz_gsm", float("nan"))
    d_dst_3h = swpc_lags[0].get("dst", float("nan")) - swpc_lags[3].get("dst", float("nan"))
    d_ey_3h = swpc_lags[0].get("ey_mvm", float("nan")) - swpc_lags[3].get("ey_mvm", float("nan"))

    # Logic ngày vs đêm (UTC+7)
    if 6 <= now_vn.hour < 19:
        mode_label = "Dự báo Sớm cho Khung Tối Nay (19:00–24:00 LT)"
        is_daytime = True
        target_slots = []
        for utc_h in range(12, 18):  # 19h..24h LT
            lt_h = (utc_h + 7) % 24
            lead = max(1, min(6, utc_h - now_utc.hour)) if now_utc.hour < utc_h else 6
            target_slots.append((utc_h, lt_h, lead))
    else:
        mode_label = "Nowcast Trực Tiếp (+1h .. +6h)"
        is_daytime = False
        target_slots = []
        for lead in range(1, 7):
            t_target = now_vn + timedelta(hours=lead)
            utc_h = t_target.astimezone(timezone.utc).hour
            lt_h = t_target.hour
            target_slots.append((utc_h, lt_h, lead))

    rows = []
    shap_peak = None
    best_vn_prob = -1.0

    for tgt in ("vn", "north", "central", "south"):
        for utc_h, lt_h, lead in target_slots:
            m_path = ROOT / "models" / f"{tgt}_lead{lead}h.joblib"
            if not m_path.exists():
                continue
            bundle = joblib.load(m_path)
            feat_cols = bundle["feat_cols"]
            row_dict = {
                "utc_hour": utc_h,
                "sin_doy": sin_doy,
                "cos_doy": cos_doy,
                "cos_2doy": cos_2doy,
                "ey_int_aft": ey_pos_sum,
                "dBz_3h": d_bz_3h,
                "dDst_3h": d_dst_3h,
                "dEy_3h": d_ey_3h,
                "dEIA_3h": float("nan"),
                "ey_x_cos2doy": float(swpc_vals.get("ey_mvm", float("nan"))) * cos_2doy,
                "bz_x_cos2doy": float(swpc_vals.get("bz_gsm", float("nan"))) * cos_2doy,
            }
            for c in feat_cols:
                if c in row_dict:
                    continue
                if "_t_minus_" in c:
                    base_c, lag_str = c.split("_t_minus_")
                    lag_h = int(lag_str.rstrip("h"))
                    # Đổi lag từ mốc target sang lag lùi từ now_utc (không vượt quá hiện tại)
                    rel_lag = max(0, min(6, lag_h - lead))
                else:
                    base_c = c
                    rel_lag = 0

                if base_c in swpc_lags.get(rel_lag, {}):
                    row_dict[c] = swpc_lags[rel_lag][base_c]
                elif base_c in swpc_vals:
                    row_dict[c] = swpc_vals[base_c]
                elif base_c in ionex_vals:
                    row_dict[c] = ionex_vals[base_c]
                elif c in igs_vals:
                    row_dict[c] = igs_vals[c]
                elif base_c in igs_vals:
                    row_dict[c] = igs_vals[base_c]
                else:
                    row_dict[c] = np.nan

            X_df = pd.DataFrame([row_dict])[feat_cols]
            prob = float(bundle["calibrated_model"].predict_proba(X_df)[0, 1])
            rows.append({
                "target": tgt,
                "utc_hour": utc_h,
                "lt_hour": lt_h,
                "lt_label": f"{lt_h:02d}:00 LT",
                "lead": lead,
                "model": bundle["model_name"],
                "prob": prob,
                "threshold": float(bundle["threshold"]),
                "data_confidence": data_confidence,
            })
            if tgt == "vn" and prob > best_vn_prob:
                best_vn_prob = prob
                shap_peak = (_shap_for_row(bundle, X_df), lt_h, lead, prob)

    fc_df = pd.DataFrame(rows)
    if save_log:
        _append_forecast_log(now_utc, mode_label, data_confidence, swpc_vals, fc_df)

    return {
        "now_vn": now_vn.strftime("%Y-%m-%d %H:%M LT"),
        "is_daytime": is_daytime,
        "mode_label": mode_label,
        "swpc": swpc_vals,
        "swpc_lags": swpc_lags,
        "ionex": ionex_vals,
        "igs": igs_vals,
        "data_confidence": data_confidence,
        "freshness": freshness,
        "forecast_df": fc_df,
        "shap_peak": shap_peak,
    }


if __name__ == "__main__":
    res = run_live_forecast()
    print(f"Time: {res['now_vn']} | Mode: {res['mode_label']} | Data Confidence: {res['data_confidence']}%")
    print("Freshness:", res["freshness"])
    print("SWPC latest:", {k: round(v, 2) if isinstance(v, float) and np.isfinite(v) else v for k, v in res["swpc"].items()})
    print("SWPC lag[0..2] Bz:", {k: round(res["swpc_lags"][k].get("bz_gsm", float("nan")), 2) for k in (0, 1, 2)})
    print(res["forecast_df"].to_string(index=False))
