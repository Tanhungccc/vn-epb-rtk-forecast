"""Tải chỉ số hoạt động Mặt Trời / bão từ từ NOAA SWPC (zero-auth, JSON)."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# Cho phép in tiếng Việt trên Windows console cp1252
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def fetch_kp_1m() -> pd.DataFrame:
    """Kp mỗi 1 phút (Wing Kp), chuỗi 1 ngày."""
    r = requests.get(config.SWPC_KP_1M, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df["time"] = pd.to_datetime(df["time_tag"], utc=True)
    # cột chuẩn: kp_index (int 0-9); nếu thiếu thì dùng estimated_kp
    if "kp_index" in df.columns:
        df["kp"] = pd.to_numeric(df["kp_index"], errors="coerce")
    elif "estimated_kp" in df.columns:
        df["kp"] = pd.to_numeric(df["estimated_kp"], errors="coerce")
    else:
        raise ValueError(f"Unexpected Kp JSON columns: {list(df.columns)}")
    return df[["time", "kp"]]


def fetch_bz_1d() -> pd.DataFrame:
    """Gió Mặt Trời realtime (DSCOVR/ACE 'SOLAR1'): Bt, Bz GSM, 1 phút, ~1 ngày."""
    r = requests.get(config.SWPC_MAG_1M, timeout=120)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df["time"] = pd.to_datetime(df["time_tag"], utc=True)
    for col in ("bz_gsm", "bt"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["time", "bz_gsm", "bt"]].dropna(subset=["bz_gsm", "bt"], how="all")


def save_snapshot(out_dir: Path) -> dict:
    """Tải snapshot hiện tại, lưu CSV, trả về thống kê nhanh."""
    out_dir.mkdir(parents=True, exist_ok=True)
    kp = fetch_kp_1m()
    bz = fetch_bz_1d()
    kp.to_csv(out_dir / "kp_1m_last7days.csv", index=False)
    bz.to_csv(out_dir / "bz_rtsw_1m.csv", index=False)
    stats = {
        "kp_now": float(kp["kp"].iloc[-1]),
        "kp_max_7d": float(kp["kp"].max()),
        "bz_now": float(bz.bz_gsm.dropna().iloc[-1]) if bz.bz_gsm.notna().any() else None,
        "bz_min_24h": float(bz.bz_gsm.min()) if bz.bz_gsm.notna().any() else None,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "summary.json").write_text(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "raw" / "swpc"
    stats = save_snapshot(out)
    print(json.dumps(stats, indent=2))
