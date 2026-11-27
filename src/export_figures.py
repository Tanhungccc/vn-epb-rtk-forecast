"""Xuất tự động bộ biểu đồ chuẩn bài báo (300 DPI PNG) vào reports/figures/.

Gồm 5 hình chính:
  1. fig1_pr_curves_vn.png          — Đường cong Precision-Recall trên tập kiểm thử mù 2026 (vn, 1h/3h/4h)
  2. fig2_calibration_reliability.png — Reliability Diagram (hiệu chuẩn xác suất, Brier Score)
  3. fig3_treeshap_1h_vs_6h.png      — TreeSHAP mean |SHAP| tại lead=1h vs lead=6h
  4. fig4_2026_blind_cases.png       — Dự báo 2026 vs thực đo COSMIC-2 S4 trên đêm bão/EPB và đêm tĩnh
  5. fig5_keograms_gannon.png        — Keogram Lon–Time & Lat–Time cho bão Gannon 2024-05-10
"""
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import fetch_madrigal, roti, train_models  # noqa: E402


def _eval_2026_probs(df: pd.DataFrame, target: str, lead: int, s4_thresh: float = 0.50):
    bundle = joblib.load(ROOT / "models" / f"{target}_lead{lead}h.joblib")
    pts_col, s4_col = train_models.COL_MAP[target]
    work, _ = train_models.build_lagged(df, lead, s4_col)
    masked = work[(work["utc_hour"] >= 12) & (work["utc_hour"] <= 17) & (work[pts_col] > 0) & work[s4_col].notna()].copy()
    masked["y_bin"] = (masked[s4_col] >= s4_thresh).astype(int)
    tr = masked[masked["year"] < 2026].copy()
    te = masked[masked["year"] >= 2026].copy()
    prob = bundle["calibrated_model"].predict_proba(te[bundle["feat_cols"]])[:, 1]

    f107_c = f"f107_t_minus_{lead}h"
    tr["hi_f107"] = (tr[f107_c].fillna(150) >= 150).astype(int)
    te["hi_f107"] = (te[f107_c].fillna(150) >= 150).astype(int)
    clim_tbl = tr.groupby(["utc_hour", "hi_f107"])["y_bin"].mean().rename("prob").reset_index()
    p_clim = pd.merge(te[["utc_hour", "hi_f107"]], clim_tbl, on=["utc_hour", "hi_f107"], how="left")["prob"].fillna(tr["y_bin"].mean()).to_numpy()
    return te, te["y_bin"].to_numpy(), prob, p_clim, bundle


def make_pr_and_calibration(df: pd.DataFrame, out_dir: Path):
    # Fig 1: PR Curves
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, lead in zip(axes, (1, 3, 4)):
        _, y_true, p_ml, p_clim, bundle = _eval_2026_probs(df, "vn", lead)
        prec_m, rec_m, _ = precision_recall_curve(y_true, p_ml)
        prec_c, rec_c, _ = precision_recall_curve(y_true, p_clim)
        ap_m = average_precision_score(y_true, p_ml)
        ap_c = average_precision_score(y_true, p_clim)
        ax.plot(rec_m, prec_m, lw=2.2, label=f"{bundle['model_name'].upper()} (AP={ap_m:.3f})")
        ax.plot(rec_c, prec_c, lw=1.8, ls="--", label=f"Climatology (AP={ap_c:.3f})")
        ax.axhline(y_true.mean(), color="gray", ls=":", lw=1.2, label=f"Base rate ({y_true.mean():.2f})")
        ax.set_title(f"Lead = +{lead}h (2026 Blind Test)")
        ax.set_xlabel("Recall")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("Precision")
    fig.suptitle("Precision–Recall Curves — Vietnam (6°–24°N, 101°–116°E, S4 ≥ 0.50)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_pr_curves_vn.png", dpi=300)
    plt.close(fig)

    # Fig 2: Calibration Reliability Diagram
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    for ax, lead in zip(axes, (1, 4)):
        _, y_true, p_ml, p_clim, bundle = _eval_2026_probs(df, "vn", lead)
        fop_m, mpv_m = calibration_curve(y_true, p_ml, n_bins=6, strategy="quantile")
        fop_c, mpv_c = calibration_curve(y_true, p_clim, n_bins=6, strategy="quantile")
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax.plot(mpv_m, fop_m, "o-", lw=2, label=f"{bundle['model_name'].upper()} (Brier={bundle['test_metrics']['brier']:.3f})")
        ax.plot(mpv_c, fop_c, "s--", lw=1.6, label="Climatology")
        ax.set_title(f"Reliability Diagram — Lead +{lead}h")
        ax.set_xlabel("Predicted Probability")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Observed Fraction of EPB (S4 ≥ 0.50)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_calibration_reliability.png", dpi=300)
    plt.close(fig)


def make_treeshap(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, lead in zip(axes, (1, 6)):
        te, _, _, _, bundle = _eval_2026_probs(df, "vn", lead)
        base = bundle["base_model"]
        cols = bundle["feat_cols"]
        if hasattr(base, "get_booster"):
            contribs = base.get_booster().predict(xgb.DMatrix(te[cols]), pred_contribs=True)[:, :-1]
        else:
            contribs = base.predict(te[cols], pred_contrib=True)[:, :-1]
        mean_abs = pd.Series(np.abs(contribs).mean(axis=0), index=cols).sort_values(ascending=True).tail(10)
        ax.barh(mean_abs.index, mean_abs.values, color="#1f77b4" if lead == 1 else "#d62728")
        ax.set_title(f"TreeSHAP Top-10 — Lead +{lead}h ({bundle['model_name'].upper()})")
        ax.set_xlabel("Mean |SHAP value| (log-odds)")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Physical Regime Shift: Local IGS ROTI (1h) → Pre-Sunset Precursors (6h)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_treeshap_1h_vs_6h.png", dpi=300)
    plt.close(fig)


def make_2026_cases(df: pd.DataFrame, out_dir: Path):
    te, _, p_ml, _, bundle = _eval_2026_probs(df, "vn", 1)
    by_day = te.groupby("date")["c2_vn_s4_max"].agg(["max", "count"])
    epb_day = by_day[by_day["max"] >= 0.6].index[0]
    quiet_day = by_day[by_day["max"] < 0.35].index[0] if (by_day["max"] < 0.35).any() else by_day["max"].idxmin()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, ds, title in zip(axes, [epb_day, quiet_day], ["2026 Active EPB Night", "2026 Quiet Night"]):
        work, _ = train_models.build_lagged(df[df["date"] == ds], 1, "c2_vn_s4_max")
        eve = work[(work["utc_hour"] >= 12) & (work["utc_hour"] <= 17)].copy()
        probs = bundle["calibrated_model"].predict_proba(eve[bundle["feat_cols"]])[:, 1]
        hrs = [f"{h:02d}Z" for h in eve["utc_hour"]]
        ax.bar(hrs, probs, color=["crimson" if p >= bundle["threshold"] else "seagreen" for p in probs], alpha=0.75, label="Pred EPB Prob (+1h)")
        ax.axhline(bundle["threshold"], color="red", ls="--", lw=1.2, label=f"Threshold ({bundle['threshold']:.2f})")
        ax2 = ax.twinx()
        m = eve["c2_vn_s4_max"].notna()
        if m.any():
            ax2.plot(np.array(hrs)[m], eve.loc[m, "c2_vn_s4_max"], "co-", lw=2, ms=7, mec="k", label="Obs S4 max")
        ax2.set_ylim(0, 1.1)
        ax2.set_ylabel("Observed S4 max")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{title}: {ds}")
        ax.set_xlabel("UTC Hour (+7h = VN LT)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Predicted EPB Probability")
    fig.suptitle("2026 Prospective Blind-Test Case Studies (Vietnam, Lead +1h)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_2026_blind_cases.png", dpi=300)
    plt.close(fig)


def make_gannon_keograms(out_dir: Path):
    p3500 = ROOT / "data" / "raw" / "tec" / "gps240510.hdf5"
    if not p3500.exists():
        return
    df_r = roti.compute_roti(roti.compute_rot(fetch_madrigal.load_gridded_tec(p3500)))
    df_r["utc_hour"] = df_r["time"].dt.hour
    sub = df_r[(df_r.utc_hour >= 12) & (df_r.utc_hour <= 17) & df_r.gdlat.between(0, 28) & df_r.glon.between(95, 122)]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    lon_t = sub.groupby(["glon", "time"])["roti"].quantile(0.95).unstack()
    lat_t = sub.groupby(["gdlat", "time"])["roti"].quantile(0.95).unstack()

    im1 = axes[0].imshow(lon_t.values, aspect="auto", origin="lower", cmap="YlOrRd", vmin=0, vmax=1.5,
                         extent=[12, 18, lon_t.index.min(), lon_t.index.max()])
    axes[0].set_title("Lon–Time Keogram (2024-05-10 Gannon Storm)")
    axes[0].set_xlabel("UTC Hour")
    axes[0].set_ylabel("Longitude (°E)")
    fig.colorbar(im1, ax=axes[0], label="ROTI p95 (TECU/min)")

    im2 = axes[1].imshow(lat_t.values, aspect="auto", origin="lower", cmap="YlOrRd", vmin=0, vmax=1.5,
                         extent=[12, 18, lat_t.index.min(), lat_t.index.max()])
    axes[1].axhline(9.0, color="cyan", ls="--", lw=1.8, label="Dip Equator (~9°N)")
    axes[1].set_title("Lat–Time Keogram (Dip Equator → EIA)")
    axes[1].set_xlabel("UTC Hour")
    axes[1].set_ylabel("Latitude (°N)")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.colorbar(im2, ax=axes[1], label="ROTI p95 (TECU/min)")

    fig.tight_layout()
    fig.savefig(out_dir / "fig5_keograms_gannon.png", dpi=300)
    plt.close(fig)


def main():
    out_dir = ROOT / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(ROOT / "data" / "dataset" / "hourly_features.parquet")
    make_pr_and_calibration(df, out_dir)
    make_treeshap(df, out_dir)
    make_2026_cases(df, out_dir)
    make_gannon_keograms(out_dir)
    for f in sorted(out_dir.glob("*.png")):
        print(f"Exported: {f.name} ({f.stat().st_size/1e3:.1f} KB)")


if __name__ == "__main__":
    main()
