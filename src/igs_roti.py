"""Tính ROTI chuẩn (Pi et al., 1997) từ file RINEX 3 30 giây (.crx.gz) của các trạm IGS.

Phương pháp:
  - Giải nén Hatanaka (.crx.gz -> RINEX 3 ASCII) bằng `hatanaka.decompress`.
  - Trích xuất pha sóng mang GPS (G): ưu tiên L1C + (L2W | L2C | L2X | L2P).
  - Tổ hợp không phụ thuộc hình học (Geometry-Free):
      sTEC_rel = 9.5196 * (lambda1 * L1 - lambda2 * L2)  [TECU]
    (sai số hằng số DCB & bội nguyên bước sóng tự triệt tiêu khi lấy sai phân ROT).
  - ROT(t) = (sTEC(t) - sTEC(t-30s)) / 0.5  [TECU/phút], lọc cycle-slip (|ROT| <= 20 TECU/phút).
  - ROTI = std(ROT) trên cửa sổ trượt 5 phút (10 mẫu 30s, tối thiểu 5 mẫu hợp lệ).
"""
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import hatanaka
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

C_LIGHT = 299_792_458.0
F1 = 1_575_420_000.0
F2 = 1_227_600_000.0
LAM1 = C_LIGHT / F1
LAM2 = C_LIGHT / F2
TECU_FACTOR = (F1**2 * F2**2) / (40.3e16 * (F1**2 - F2**2))  # ~9.5196 TECU/m


def ecef_to_geodetic(x: float, y: float, z: float) -> tuple[float, float]:
    """Chuyển ECEF (m) sang (lat_deg, lon_deg) WGS84."""
    a = 6378137.0
    e2 = 6.69437999014e-3
    lon = math.degrees(math.atan2(y, x))
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - e2))
    for _ in range(5):
        n = a / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
        lat = math.atan2(z + e2 * n * math.sin(lat), p)
    return math.degrees(lat), lon


def _pick_obs_indices(obs_types: list[str]) -> tuple[int, int] | None:
    l1_cands = ["L1C", "L1W", "L1X", "L1P"]
    l2_cands = ["L2W", "L2C", "L2X", "L2P", "L2S", "L2L"]
    i1 = next((obs_types.index(c) for c in l1_cands if c in obs_types), None)
    i2 = next((obs_types.index(c) for c in l2_cands if c in obs_types), None)
    if i1 is None or i2 is None:
        return None
    return i1, i2


def parse_rinex3_gps_phase(crx_gz_path: Path) -> tuple[dict, pd.DataFrame]:
    """Đọc file .crx.gz, trả về (station_meta, DataFrame[time, sat, l1, l2])."""
    txt = hatanaka.decompress(crx_gz_path.read_bytes()).decode("ascii", errors="ignore")
    lines = txt.splitlines()

    lat_r, lon_r = float("nan"), float("nan")
    obs_G: list[str] = []
    idx = 0
    # Đọc header
    while idx < len(lines):
        ln = lines[idx]
        idx += 1
        tag = ln[60:].strip() if len(ln) > 60 else ""
        if tag == "APPROX POSITION XYZ":
            parts = ln[:60].split()
            if len(parts) >= 3:
                lat_r, lon_r = ecef_to_geodetic(float(parts[0]), float(parts[1]), float(parts[2]))
        elif tag == "SYS / # / OBS TYPES":
            if ln[0] == "G":
                n_obs = int(ln[3:6])
                obs_G.extend(ln[7:60].split())
                while len(obs_G) < n_obs and idx < len(lines):
                    obs_G.extend(lines[idx][7:60].split())
                    idx += 1
        elif tag == "END OF HEADER":
            break

    pair = _pick_obs_indices(obs_G)
    if pair is None:
        raise ValueError(f"Cannot find dual-freq GPS L1/L2 in {crx_gz_path.name}: {obs_G}")
    i1, i2 = pair
    max_idx = max(i1, i2)

    times, sats, l1s, l2s = [], [], [], []
    cur_t = None
    while idx < len(lines):
        ln = lines[idx]
        idx += 1
        if not ln:
            continue
        if ln[0] == ">":
            # > YYYY MM DD HH MM SS.SSSSSSS  flag  nsat
            parts = ln[1:].split()
            if len(parts) < 8:
                continue
            yr, mo, dy, hr, mn = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            sc = int(float(parts[5]))
            flag = int(parts[6])
            nsat = int(parts[7])
            if flag not in (0, 1):
                idx += nsat
                continue
            cur_t = datetime(yr, mo, dy, hr, mn, sc, tzinfo=timezone.utc)
            for _ in range(nsat):
                if idx >= len(lines):
                    break
                sln = lines[idx]
                idx += 1
                if not sln.startswith("G"):
                    continue
                sat = sln[:3]
                # Mỗi trường obs rộng 16 ký tự (14 số + LLI + SSI) bắt đầu từ col 3
                p1 = 3 + i1 * 16
                p2 = 3 + i2 * 16
                if len(sln) < max(p1, p2) + 14:
                    continue
                s1 = sln[p1 : p1 + 14].strip()
                s2 = sln[p2 : p2 + 14].strip()
                if not s1 or not s2:
                    continue
                try:
                    v1, v2 = float(s1), float(s2)
                except ValueError:
                    continue
                times.append(cur_t)
                sats.append(sat)
                l1s.append(v1)
                l2s.append(v2)

    df = pd.DataFrame({"time": times, "sat": sats, "l1": l1s, "l2": l2s})
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    meta = {
        "station": crx_gz_path.name[:4],
        "lat": lat_r,
        "lon": lon_r,
        "l1_obs": obs_G[i1],
        "l2_obs": obs_G[i2],
    }
    return meta, df


def compute_igs_roti(df: pd.DataFrame, max_rot: float = 20.0) -> pd.DataFrame:
    """Tính ROT (TECU/phút) và ROTI 5-phút (std trên 10 mẫu 30s, tối thiểu 5 mẫu)."""
    if df.empty:
        return df
    df = df.sort_values(["sat", "time"]).copy()
    df["stec_rel"] = TECU_FACTOR * (LAM1 * df["l1"] - LAM2 * df["l2"])
    prev_stec = df.groupby("sat")["stec_rel"].shift(1)
    prev_t = df.groupby("sat")["time"].shift(1)
    dt_sec = (df["time"] - prev_t).dt.total_seconds()
    rot = np.where((dt_sec >= 25.0) & (dt_sec <= 35.0) & prev_stec.notna(),
                   (df["stec_rel"] - prev_stec) / (dt_sec / 60.0), np.nan)
    rot = np.where(np.abs(rot) <= max_rot, rot, np.nan)
    df["rot"] = rot
    df["roti"] = (
        df.groupby("sat")["rot"]
        .transform(lambda s: s.rolling(window=10, min_periods=5).std(ddof=1))
    )
    df["time_5min"] = df["time"].dt.floor("5min")
    df["utc_hour"] = df["time"].dt.hour
    return df
