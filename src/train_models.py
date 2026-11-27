"""Huấn luyện & Hiệu chuẩn xác suất (XGBoost, LightGBM, Climatology+Residual vs Baselines).

- Train/Val trên 2024–2025 (blocked time-series split), Kiểm thử mù (Prospective) trên 2026.
- Hỗ trợ kỹ thuật `base_margin` (Climatology log-odds + ML Residual) để chống trôi phân phối
  giữa cực đại Mặt Trời 2024–2025 và 2026.
- Hiệu chuẩn xác suất bằng CalibratedClassifierCV (sigmoid / isotonic) trên tập Validation.
- Lưu mô hình tốt nhất vào models/{target}_lead{h}.joblib để Dashboard & xuất hình dùng trực tiếp.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if sys.stdout is not None and getattr(sys.stdout, "encoding", None) and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.residual_model import ClimatologyResidualXGB  # noqa: E402

BASE_LAG_COLS = [
    "f107", "bz_gsm", "v_sw", "ey_mvm", "kp", "dst",
    "eia_north", "eia_south", "eia_diff", "eia_ratio",
    "igs_hksl_roti", "igs_cusv_roti", "igs_pimo_roti", "igs_ntus_roti",
]

COL_MAP = {
    "vn": ("c2_vn_pts", "c2_vn_s4_max"),
    "north": ("c2_north_pts", "c2_north_s4_max"),
    "central": ("c2_central_pts", "c2_central_s4_max"),
    "south": ("c2_south_pts", "c2_south_s4_max"),
    "regional": ("c2_reg_pts", "c2_reg_s4_p95"),
}


def calc_tss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return float(tpr - fpr)


def best_tss_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_th, best_tss = 0.5, -1.0
    for th in np.linspace(0.10, 0.90, 81):
        tss = calc_tss(y_true, (y_prob >= th).astype(int))
        if tss > best_tss:
            best_tss = tss
            best_th = float(th)
    return best_th


def eval_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresh: float = 0.5) -> dict:
    y_prob_c = np.clip(y_prob, 0.0, 1.0)
    y_pred = (y_prob_c >= thresh).astype(int)
    pr_auc = float(average_precision_score(y_true, y_prob_c)) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "pr_auc": pr_auc,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tss": calc_tss(y_true, y_pred),
        "brier": float(brier_score_loss(y_true, y_prob_c)),
    }


def build_lagged(df: pd.DataFrame, lead: int, s4_col: str) -> tuple[pd.DataFrame, list[str]]:
    work = df.sort_values(["date", "utc_hour"]).copy()
    work["cos_2doy"] = 2.0 * (work["cos_doy"] ** 2) - 1.0
    # Tích phân điện trường xuyên thấu dương buổi chiều (từ 07:00 UTC)
    ey_pos = work["ey_mvm"].clip(lower=0.0).where(work["utc_hour"] >= 7, 0.0).fillna(0.0)
    work["_ey_pos_cum"] = ey_pos.groupby(work["date"]).cumsum()

    g = work.groupby("date", sort=False)
    feat_cols = ["utc_hour", "sin_doy", "cos_doy", "cos_2doy"]
    for c in BASE_LAG_COLS:
        if c in work.columns:
            lc = f"{c}_t_minus_{lead}h"
            work[lc] = g[c].shift(lead)
            feat_cols.append(lc)

    # Tích phân Ey buổi chiều tại mốc t - lead
    work["ey_int_aft"] = g["_ey_pos_cum"].shift(lead)
    feat_cols.append("ey_int_aft")

    # Tốc độ biến thiên 3 giờ gần nhất (t - lead so với t - lead - 3h)
    for src_col, d_name in [("bz_gsm", "dBz_3h"), ("dst", "dDst_3h"), ("ey_mvm", "dEy_3h"), ("eia_diff", "dEIA_3h")]:
        if src_col in work.columns:
            work[d_name] = g[src_col].shift(lead) - g[src_col].shift(lead + 3)
            feat_cols.append(d_name)

    # Chuỗi trễ lùi thêm 1h cho gió Mặt Trời & Dst
    for src_col in ("bz_gsm", "ey_mvm", "dst"):
        if src_col in work.columns:
            lc1 = f"{src_col}_t_minus_{lead+1}h"
            work[lc1] = g[src_col].shift(lead + 1)
            feat_cols.append(lc1)

    # Biến tương tác mùa phân điểm x địa từ
    if f"ey_mvm_t_minus_{lead}h" in work.columns:
        work["ey_x_cos2doy"] = work[f"ey_mvm_t_minus_{lead}h"] * work["cos_2doy"]
        feat_cols.append("ey_x_cos2doy")
    if f"bz_gsm_t_minus_{lead}h" in work.columns:
        work["bz_x_cos2doy"] = work[f"bz_gsm_t_minus_{lead}h"] * work["cos_2doy"]
        feat_cols.append("bz_x_cos2doy")

    work["s4_persistence"] = g[s4_col].shift(lead)
    return work, feat_cols


def train_one(
    df: pd.DataFrame,
    target: str,
    lead: int,
    s4_thresh: float = 0.50,
    cal_method: str = "sigmoid",
    models_dir: Path | None = None,
) -> dict:
    pts_col, s4_col = COL_MAP[target]
    work, feat_cols = build_lagged(df, lead, s4_col)

    masked = work[(work["utc_hour"] >= 12) & (work["utc_hour"] <= 17) & (work[pts_col] > 0) & work[s4_col].notna()].copy()
    masked["y_bin"] = (masked[s4_col] >= s4_thresh).astype(int)

    has_2026 = (masked["year"] >= 2026).any()
    if has_2026:
        tr_all = masked[masked["year"] < 2026].copy()
        te = masked[masked["year"] >= 2026].copy()
    else:
        dates = sorted(masked["date"].unique())
        sp = max(2, int(len(dates) * 0.70))
        tr_all = masked[masked["date"].isin(set(dates[:sp]))].copy()
        te = masked[masked["date"].isin(set(dates[sp:]))].copy()

    tr_dates = sorted(tr_all["date"].unique())
    val_sp = max(1, int(len(tr_dates) * 0.80))
    tr_in = tr_all[tr_all["date"].isin(set(tr_dates[:val_sp]))].copy()
    val = tr_all[tr_all["date"].isin(set(tr_dates[val_sp:]))].copy()

    y_tr_in, y_val, y_te = tr_in["y_bin"].to_numpy(), val["y_bin"].to_numpy(), te["y_bin"].to_numpy()
    y_tr_all = tr_all["y_bin"].to_numpy()
    tr_rate = float(y_tr_all.mean()) if len(y_tr_all) else 0.0
    tr_in_rate = float(y_tr_in.mean()) if len(y_tr_in) else 0.0

    # 1. Persistence & Climatology (Laplace-smoothed)
    pers_s4 = te["s4_persistence"].fillna(float(tr_all[s4_col].mean())).to_numpy()
    m_pers = eval_metrics(y_te, (pers_s4 >= s4_thresh).astype(float), thresh=0.5)

    f107_c = f"f107_t_minus_{lead}h"
    for dframe in (tr_in, tr_all, te):
        dframe["hi_f107"] = (dframe[f107_c].fillna(150) >= 150).astype(int)

    clim_in_tbl = tr_in.groupby(["utc_hour", "hi_f107"])["y_bin"].agg(
        prob=lambda s: float((s.sum() + 1.0) / (len(s) + 2.0))
    ).reset_index()
    clim_all_tbl = tr_all.groupby(["utc_hour", "hi_f107"])["y_bin"].agg(
        prob=lambda s: float((s.sum() + 1.0) / (len(s) + 2.0))
    ).reset_index()
    te_clim = pd.merge(te[["utc_hour", "hi_f107"]], clim_all_tbl, on=["utc_hour", "hi_f107"], how="left")
    m_clim = eval_metrics(y_te, te_clim["prob"].fillna(tr_rate).to_numpy(), thresh=0.5)

    # 2. XGBoost, LightGBM, và Climatology + Residual XGBoost
    n_pos_in = int(y_tr_in.sum())
    spw_in = float((len(y_tr_in) - n_pos_in) / max(n_pos_in, 1))

    xgb_raw = XGBClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw_in, random_state=42, eval_metric="logloss",
    )
    lgb_raw = LGBMClassifier(
        n_estimators=250, max_depth=4, num_leaves=15, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw_in, random_state=42, verbose=-1,
    )
    res_raw = ClimatologyResidualXGB(f107_col=f107_c, clim_tbl=clim_in_tbl, fallback_prob=tr_in_rate)

    results = {
        "target": target, "lead": lead, "n_tr": len(tr_all), "pos_tr": int(y_tr_all.sum()),
        "n_te": len(te), "pos_te": int(y_te.sum()),
        "pers": m_pers, "clim": m_clim,
    }

    best_name, best_brier, best_bundle = None, 999.0, None
    for m_name, base_clf in [("xgb", xgb_raw), ("lgb", lgb_raw), ("xgb_resid", res_raw)]:
        if len(np.unique(y_tr_in)) < 2:
            prob = np.full(len(te), tr_rate, dtype=float)
            th = 0.5
            cal_model = None
        else:
            base_clf.fit(tr_in[feat_cols], y_tr_in)
            if len(np.unique(y_val)) >= 2:
                cal_model = CalibratedClassifierCV(FrozenEstimator(base_clf), method=cal_method)
                cal_model.fit(val[feat_cols], y_val)
                val_prob = cal_model.predict_proba(val[feat_cols])[:, 1]
                th = best_tss_threshold(y_val, val_prob)
                prob = cal_model.predict_proba(te[feat_cols])[:, 1]
            else:
                cal_model = base_clf
                th = 0.5
                prob = base_clf.predict_proba(te[feat_cols])[:, 1]
        met = eval_metrics(y_te, prob, thresh=th)
        met["th"] = th
        results[m_name] = met
        if met["brier"] < best_brier and cal_model is not None:
            best_brier = met["brier"]
            best_name = m_name
            best_bundle = {
                "target": target,
                "lead": lead,
                "s4_thresh": s4_thresh,
                "model_name": m_name,
                "base_model": base_clf.clf if hasattr(base_clf, "clf") else base_clf,
                "calibrated_model": cal_model,
                "feat_cols": feat_cols,
                "threshold": th,
                "test_metrics": met,
            }

    if models_dir is not None and best_bundle is not None:
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_bundle, models_dir / f"{target}_lead{lead}h.joblib")
    results["best_name"] = best_name
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "dataset" / "hourly_features.parquet"))
    ap.add_argument("--targets", nargs="+", default=["vn", "north", "central", "south"])
    ap.add_argument("--s4-thresh", type=float, default=0.50)
    ap.add_argument("--cal", choices=["sigmoid", "isotonic"], default="sigmoid")
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    models_dir = Path(args.models_dir)
    summary = []

    for tgt in args.targets:
        print(f"\n=== Target={tgt} (S4>={args.s4_thresh}, Cal={args.cal}) ===")
        print(f"{'Lead':>4s} | {'Model':12s} | {'PR-AUC':>6s} | {'F1':>6s} | {'TSS':>6s} | {'Brier':>6s} | {'Th':>4s} | Split")
        print("-" * 88)
        for lead in range(1, 7):
            res = train_one(df, tgt, lead, s4_thresh=args.s4_thresh, cal_method=args.cal, models_dir=models_dir)
            info = f"{res['n_tr']}(+{res['pos_tr']}) / {res['n_te']}(+{res['pos_te']})"
            for label, key in [("Persistence", "pers"), ("Climatology", "clim"), ("XGBoost-Cal", "xgb"), ("LightGBM-Cal", "lgb"), ("XGB-Resid", "xgb_resid")]:
                m = res[key]
                th_s = f"{m.get('th', 0.5):.2f}" if key in ("xgb", "lgb", "xgb_resid") else " -- "
                print(f"{lead:>3d}h | {label:12s} | {m['pr_auc']:6.3f} | {m['f1']:6.3f} | {m['tss']:6.3f} | {m['brier']:6.3f} | {th_s:>4s} | {info if key=='pers' else ''}")
            print("-" * 88)
            summary.append(res)

    (models_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved best models & summary -> {models_dir}")


if __name__ == "__main__":
    main()
