"""Pipeline tái huấn luyện định kỳ 30 ngày theo chuẩn Champion vs. Challenger.

Quy trình:
  1. Gộp data/dataset/hourly_features.parquet + data/dataset/live_accumulated.parquet (nếu có).
  2. Tách 30 ngày gần nhất làm tập đối đầu công bằng (Evaluation Window).
  3. Huấn luyện bộ mô hình ứng viên (Challenger) trên dữ liệu trước cửa sổ 30 ngày.
  4. Chấm điểm Champion (models/*.joblib) vs. Challenger trên cùng 30 ngày gần nhất
     theo PR-AUC, TSS và Brier Score.
  5. Chỉ ghi đè models/{target}_lead{h}.joblib nếu Challenger thắng điểm
     (Brier tốt hơn hoặc PR-AUC & TSS cải thiện mà Brier không xấu đi), đồng thời
     backup bản Champion cũ vào models/archive/<timestamp>/.

Chạy thử (không ghi đè models/):
  python -m src.retrain_pipeline --dry-run
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import train_models  # noqa: E402

BASE_DATA = ROOT / "data" / "dataset" / "hourly_features.parquet"
LIVE_DATA = ROOT / "data" / "dataset" / "live_accumulated.parquet"
MODELS_DIR = ROOT / "models"


def load_merged_dataset() -> pd.DataFrame:
    df = pd.read_parquet(BASE_DATA)
    if LIVE_DATA.exists():
        df_live = pd.read_parquet(LIVE_DATA)
        if not df_live.empty:
            df = pd.concat([df, df_live], ignore_index=True)
            df = df.drop_duplicates(subset=["date", "utc_hour"], keep="last")
    return df.sort_values(["date", "utc_hour"]).reset_index(drop=True)


def _eval_bundle_on_df(bundle: dict, df_eval: pd.DataFrame, target: str, lead: int) -> dict | None:
    pts_col, s4_col = train_models.COL_MAP[target]
    work, _ = train_models.build_lagged(df_eval, lead, s4_col)
    sub = work[(work["utc_hour"] >= 12) & (work["utc_hour"] <= 17) & (work[pts_col] > 0) & work[s4_col].notna()].copy()
    if sub.empty or sub[s4_col].nunique() < 2:
        return None
    y_true = (sub[s4_col] >= bundle.get("s4_thresh", 0.50)).astype(int).to_numpy()
    if len(np.unique(y_true)) < 2:
        return None
    cols = bundle["feat_cols"]
    for c in cols:
        if c not in sub.columns:
            sub[c] = np.nan
    prob = bundle["calibrated_model"].predict_proba(sub[cols])[:, 1]
    return train_models.eval_metrics(y_true, prob, thresh=float(bundle.get("threshold", 0.50)))


def _challenger_wins(champ: dict, chall: dict) -> bool:
    """Challenger thắng nếu Brier tốt hơn, hoặc cả PR-AUC & TSS đều vượt mà Brier không xấu hơn > 0.005."""
    if chall["brier"] < champ["brier"] - 1e-4:
        return True
    if chall["pr_auc"] > champ["pr_auc"] and chall["tss"] > champ["tss"] and chall["brier"] <= champ["brier"] + 0.005:
        return True
    return False


def run_retrain_pipeline(eval_days: int = 30, dry_run: bool = False) -> list[dict]:
    df = load_merged_dataset()
    all_dates = sorted(df["date"].unique())
    if len(all_dates) <= eval_days + 30:
        raise ValueError(f"Không đủ ngày ({len(all_dates)}) để tách {eval_days} ngày đối đầu.")

    cutoff_dates = set(all_dates[-eval_days:])
    df_train = df[~df["date"].isin(cutoff_dates)].copy()
    df_eval = df[df["date"].isin(cutoff_dates)].copy()

    tmp_dir = MODELS_DIR / "challenger_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ts_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_dir = MODELS_DIR / "archive" / ts_tag
    report_rows = []
    promoted = 0

    print(f"Champion vs Challenger trên {eval_days} ngày gần nhất ({min(cutoff_dates)} -> {max(cutoff_dates)})")
    print(f"{'Model':14s} | {'Champ PR/TSS/Brier':22s} | {'Chall PR/TSS/Brier':22s} | Quyết định")
    print("-" * 82)

    for tgt in ("vn", "north", "central", "south"):
        for lead in range(1, 7):
            key = f"{tgt}_lead{lead}h"
            champ_path = MODELS_DIR / f"{key}.joblib"
            train_models.train_one(df_train, tgt, lead, s4_thresh=0.50, cal_method="sigmoid", models_dir=tmp_dir)
            chall_path = tmp_dir / f"{key}.joblib"
            if not chall_path.exists():
                continue

            chall_bundle = joblib.load(chall_path)
            m_chall = _eval_bundle_on_df(chall_bundle, df_eval, tgt, lead)

            if not champ_path.exists():
                decision = "NEW"
                m_champ = {"pr_auc": float("nan"), "tss": float("nan"), "brier": float("nan")}
                win = True
            else:
                champ_bundle = joblib.load(champ_path)
                m_champ = _eval_bundle_on_df(champ_bundle, df_eval, tgt, lead)
                if m_champ is None or m_chall is None:
                    win = False
                    decision = "SKIP (thiếu nhãn)"
                else:
                    win = _challenger_wins(m_champ, m_chall)
                    decision = "PROMOTE" if win else "KEEP CHAMPION"

            if win and not dry_run:
                if champ_path.exists():
                    archive_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(champ_path, archive_dir / champ_path.name)
                shutil.copy2(chall_path, champ_path)
                promoted += 1

            c_str = f"{m_champ['pr_auc']:.3f}/{m_champ['tss']:.3f}/{m_champ['brier']:.3f}" if m_champ else "N/A"
            n_str = f"{m_chall['pr_auc']:.3f}/{m_chall['tss']:.3f}/{m_chall['brier']:.3f}" if m_chall else "N/A"
            print(f"{key:14s} | {c_str:22s} | {n_str:22s} | {decision}{' (dry)' if dry_run else ''}")
            report_rows.append({
                "key": key, "decision": decision, "promoted": bool(win and not dry_run),
                "champion": m_champ, "challenger": m_chall,
            })

    shutil.rmtree(tmp_dir, ignore_errors=True)
    rep_path = MODELS_DIR / "retrain_last_report.json"
    rep_path.write_text(json.dumps(report_rows, indent=2, default=str), encoding="utf-8")
    print(f"\nHoàn tất: {promoted}/24 mô hình được nâng cấp. Báo cáo -> {rep_path}")
    return report_rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Tái huấn luyện Champion vs. Challenger định kỳ 30 ngày.")
    ap.add_argument("--eval-days", type=int, default=30, help="Số ngày gần nhất dùng để chấm điểm đối đầu.")
    ap.add_argument("--dry-run", action="store_true", help="Chỉ chạy đối đầu so sánh, không ghi đè models/*.joblib.")
    args = ap.parse_args()
    run_retrain_pipeline(eval_days=args.eval_days, dry_run=args.dry_run)
