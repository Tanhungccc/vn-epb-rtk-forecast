"""Đối chuẩn Baseline (Persistence + Climatology) vs XGBoost cho 6 tầm dự báo (lead_hours = 1..6).

Nguyên tắc khoa học:
  1. Chống rò rỉ biên ngày: Mọi phép .shift() tạo đặc trưng ở thời điểm phát (t - lead)
     đều thực hiện trong groupby("date"), tuyệt đối không trượt qua ngày khác.
  2. Blocked time-series split: Chia theo khối ngày liên tục (Train / Val / Test).
  3. Masked evaluation: Chỉ chấm điểm tại các giờ tối (12–17 UTC) có lát cắt COSMIC-2 scnLv2.
  4. Tối ưu ngưỡng phân loại theo TSS trên tập Validation (tránh cố định 0.5 khi mất cân bằng lớp).
  5. In Top-10 đặc trưng quan trọng nhất tại mốc lead = 1h và lead = 6h.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_LAG_COLS = [
    "f107", "bz_gsm", "v_sw", "ey_mvm", "kp", "dst",
    "eia_north", "eia_south", "eia_diff", "eia_ratio",
    "igs_hksl_roti", "igs_cusv_roti", "igs_pimo_roti", "igs_ntus_roti",
]


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


def _make_xgb(spw: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, random_state=42, eval_metric="logloss",
    )


def run_for_lead(df: pd.DataFrame, lead: int, pts_col: str, s4_col: str, s4_thresh: float, test_frac: float):
    work = df.sort_values(["date", "utc_hour"]).copy()
    g = work.groupby("date", sort=False)
    feat_cols = ["utc_hour", "sin_doy", "cos_doy"]
    for c in BASE_LAG_COLS:
        if c in work.columns:
            lc = f"{c}_t_minus_{lead}h"
            work[lc] = g[c].shift(lead)
            feat_cols.append(lc)

    work["s4_persistence"] = g[s4_col].shift(lead)

    masked = work[(work["utc_hour"] >= 12) & (work["utc_hour"] <= 17) & (work[pts_col] > 0) & work[s4_col].notna()].copy()
    masked["y_bin"] = (masked[s4_col] >= s4_thresh).astype(int)

    dates = sorted(masked["date"].unique())
    split_idx = max(2, int(len(dates) * (1.0 - test_frac)))
    tr_dates = dates[:split_idx]
    te_dates = set(dates[split_idx:])

    # Tách Validation từ 20% cuối của khối Train để tối ưu ngưỡng TSS
    val_split = max(1, int(len(tr_dates) * 0.80))
    tr_in_dates, val_dates = set(tr_dates[:val_split]), set(tr_dates[val_split:])

    tr = masked[masked["date"].isin(set(tr_dates))].copy()
    tr_in = masked[masked["date"].isin(tr_in_dates)].copy()
    val = masked[masked["date"].isin(val_dates)].copy()
    te = masked[masked["date"].isin(te_dates)].copy()

    y_tr, y_te = tr["y_bin"].to_numpy(), te["y_bin"].to_numpy()
    tr_rate = float(y_tr.mean()) if len(y_tr) else 0.0

    # 1. Persistence
    pers_s4 = te["s4_persistence"].fillna(float(tr[s4_col].mean())).to_numpy()
    m_pers = eval_metrics(y_te, (pers_s4 >= s4_thresh).astype(float), thresh=0.5)

    # 2. Climatology (utc_hour + F10.7 >= 150) + ngưỡng TSS từ Val
    f107_col = f"f107_t_minus_{lead}h"
    tr["hi_f107"] = (tr[f107_col].fillna(150) >= 150).astype(int)
    tr_in["hi_f107"] = (tr_in[f107_col].fillna(150) >= 150).astype(int)
    val["hi_f107"] = (val[f107_col].fillna(150) >= 150).astype(int)
    te["hi_f107"] = (te[f107_col].fillna(150) >= 150).astype(int)

    clim_val_tbl = tr_in.groupby(["utc_hour", "hi_f107"])["y_bin"].mean().rename("prob").reset_index()
    val_clim = pd.merge(val[["utc_hour", "hi_f107"]], clim_val_tbl, on=["utc_hour", "hi_f107"], how="left")["prob"].fillna(float(tr_in["y_bin"].mean())).to_numpy()
    clim_th = best_tss_threshold(val["y_bin"].to_numpy(), val_clim) if len(np.unique(val["y_bin"])) > 1 else 0.5

    clim_tbl = tr.groupby(["utc_hour", "hi_f107"])["y_bin"].mean().rename("prob").reset_index()
    te_clim = pd.merge(te[["utc_hour", "hi_f107"]], clim_tbl, on=["utc_hour", "hi_f107"], how="left")
    m_clim = eval_metrics(y_te, te_clim["prob"].fillna(tr_rate).to_numpy(), thresh=clim_th)

    # 3. XGBoost với scale_pos_weight + tối ưu ngưỡng TSS trên Validation
    n_pos = int(y_tr.sum())
    n_neg = int(len(y_tr) - n_pos)
    top_feats = None
    xgb_th = 0.5
    if len(np.unique(y_tr)) < 2:
        xgb_prob = np.full(len(te), tr_rate, dtype=float)
    else:
        y_tr_in = tr_in["y_bin"].to_numpy()
        y_val = val["y_bin"].to_numpy()
        if len(np.unique(y_tr_in)) >= 2 and len(np.unique(y_val)) >= 2:
            spw_in = float((len(y_tr_in) - y_tr_in.sum()) / max(int(y_tr_in.sum()), 1))
            clf_val = _make_xgb(spw_in)
            clf_val.fit(tr_in[feat_cols], y_tr_in)
            val_prob = clf_val.predict_proba(val[feat_cols])[:, 1]
            xgb_th = best_tss_threshold(y_val, val_prob)

        spw = float(n_neg / max(n_pos, 1))
        clf = _make_xgb(spw)
        clf.fit(tr[feat_cols], y_tr)
        xgb_prob = clf.predict_proba(te[feat_cols])[:, 1]
        top_feats = pd.Series(clf.feature_importances_, index=feat_cols).sort_values(ascending=False).head(10)

    m_xgb = eval_metrics(y_te, xgb_prob, thresh=xgb_th)
    return {
        "lead": lead,
        "n_tr": len(tr), "pos_tr": n_pos,
        "n_te": len(te), "pos_te": int(y_te.sum()),
        "xgb_th": xgb_th,
        "pers": m_pers, "clim": m_clim, "xgb": m_xgb,
        "top_feats": top_feats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "dataset" / "hourly_features.parquet"))
    ap.add_argument("--target", choices=["regional", "vn", "north", "central", "south"], default="regional")
    ap.add_argument("--s4-thresh", type=float, default=0.30)
    ap.add_argument("--test-frac", type=float, default=0.30)
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    col_map = {
        "regional": ("c2_reg_pts", "c2_reg_s4_p95"),
        "vn": ("c2_vn_pts", "c2_vn_s4_p95"),
        "north": ("c2_north_pts", "c2_north_s4_max"),
        "central": ("c2_central_pts", "c2_central_s4_max"),
        "south": ("c2_south_pts", "c2_south_s4_max"),
    }
    pts_col, s4_col = col_map[args.target]

    print(f"=== Benchmark Đa Tầm Dự Báo (target={args.target}, S4>={args.s4_thresh}, Val-TSS threshold) ===")
    print(f"{'Lead':>4s} | {'Model':11s} | {'PR-AUC':>6s} | {'F1':>6s} | {'TSS':>6s} | {'Brier':>6s} | {'Th':>4s} | Split (Train/Test)")
    print("-" * 86)
    saved_imps = {}
    for lead in range(1, 7):
        res = run_for_lead(df, lead, pts_col, s4_col, args.s4_thresh, args.test_frac)
        info = f"{res['n_tr']}(+{res['pos_tr']}) / {res['n_te']}(+{res['pos_te']})"
        for label, key in [("Persistence", "pers"), ("Climatology", "clim"), ("XGBoost", "xgb")]:
            m = res[key]
            th_s = f"{res['xgb_th']:.2f}" if key == "xgb" else " -- "
            print(f"{lead:>3d}h | {label:11s} | {m['pr_auc']:6.3f} | {m['f1']:6.3f} | {m['tss']:6.3f} | {m['brier']:6.3f} | {th_s:>4s} | {info if key=='pers' else ''}")
        print("-" * 86)
        if lead in (1, 6) and res["top_feats"] is not None:
            saved_imps[lead] = res["top_feats"]

    for lead in (1, 6):
        if lead in saved_imps:
            print(f"\nTop-10 Feature Importance (target={args.target}, lead={lead}h):")
            for fname, val in saved_imps[lead].items():
                print(f"  {fname:26s}: {val:.4f}")


if __name__ == "__main__":
    main()
