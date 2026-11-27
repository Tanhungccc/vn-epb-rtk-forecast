"""Module gửi cảnh báo Telegram Bot khi xác suất EPB miền Bắc/Trung/Nam vượt ngưỡng VÀNG/ĐỎ.

Cấu hình qua biến môi trường hoặc file .env ở thư mục gốc dự án:
  TELEGRAM_BOT_TOKEN=...
  TELEGRAM_CHAT_ID=...

Cơ chế chống spam (Cooldown 3 giờ):
  Lưu trạng thái lần gửi gần nhất vào data/alert_state.json; không gửi lặp lại
  trong vòng 3 giờ cho cùng mức cảnh báo (trừ khi mức cảnh báo tăng lên, ví dụ VÀNG -> ĐỎ).

Chạy thử (không gửi thật):
  python -m src.send_alert --dry-run
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import live_forecast  # noqa: E402

ALERT_STATE_PATH = ROOT / "data" / "alert_state.json"
COOLDOWN_SECONDS = 3 * 3600

LEVEL_RANK = {"XANH": 0, "VÀNG": 1, "ĐỎ": 2}
BAND_NAMES = {
    "north": "Bắc Bộ (18°–24°N)",
    "central": "Trung Bộ (12°–18°N)",
    "south": "Nam Bộ (6°–12°N)",
}


def load_env_file(env_path: Path | None = None) -> None:
    """Đọc file .env đơn giản (KEY=VALUE) nếu biến chưa có trong os.environ."""
    p = env_path or (ROOT / ".env")
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _risk_level(prob: float) -> str:
    if prob >= 0.65:
        return "ĐỎ"
    if prob >= 0.35:
        return "VÀNG"
    return "XANH"


def _load_state() -> dict:
    if ALERT_STATE_PATH.exists():
        try:
            return json.loads(ALERT_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_state(state: dict) -> None:
    ALERT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALERT_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def build_alert_payload(live: dict) -> tuple[str, str]:
    """Tạo (highest_level, message_text) từ kết quả run_live_forecast()."""
    fc = live.get("forecast_df")
    if fc is None or fc.empty:
        return "XANH", ""

    alerts = []
    highest = "XANH"
    for tgt in ("north", "central", "south"):
        sub = fc[fc["target"] == tgt]
        if sub.empty:
            continue
        idx_max = sub["prob"].idxmax()
        p_max = float(sub.loc[idx_max, "prob"])
        peak_lt = str(sub.loc[idx_max, "lt_label"])
        lvl = _risk_level(p_max)
        if LEVEL_RANK[lvl] > LEVEL_RANK[highest]:
            highest = lvl
        if lvl in ("VÀNG", "ĐỎ"):
            icon = "🔴" if lvl == "ĐỎ" else "🟡"
            alerts.append(f"{icon} {BAND_NAMES[tgt]}: *{lvl}* ({p_max*100:.1f}% lúc {peak_lt})")

    if not alerts:
        return "XANH", ""

    sw = live.get("swpc", {})
    conf = live.get("data_confidence", 0)
    lines = [
        f"⚠️ *CẢNH BÁO NHIỄU TẦNG ĐIỆN LY EPB / RTK-GNSS — MỨC {highest}*",
        f"🕒 {live.get('now_vn', '--')} | Chế độ: {live.get('mode_label', '--')}",
        f"📊 Độ tin cậy dữ liệu: *{conf}%*",
        "",
        *alerts,
        "",
        f"☀️ SWPC: Bz={sw.get('bz_gsm', float('nan')):.1f} nT | v={sw.get('v_sw', float('nan')):.0f} km/s | Kp={sw.get('kp', float('nan')):.1f}",
        "👉 Khuyến nghị: Kiểm tra kỹ Fix RTK và hạn chế bay UAV tầm xa vào khung giờ cao điểm.",
    ]
    return highest, "\n".join(lines)


def send_telegram_alert(dry_run: bool = False, force: bool = False) -> dict:
    """Kiểm tra dự báo hiện tại và gửi Telegram nếu vượt ngưỡng VÀNG/ĐỎ và hết cooldown 3h."""
    load_env_file()
    live = live_forecast.run_live_forecast()
    highest, msg = build_alert_payload(live)

    if highest == "XANH" or not msg:
        return {"sent": False, "reason": "Tất cả các miền đều ở mức XANH (<35%).", "level": highest}

    now_utc = datetime.now(timezone.utc)
    state = _load_state()
    last_lvl = state.get("last_level", "XANH")
    last_ts_str = state.get("last_sent_utc")

    if not force and last_ts_str:
        try:
            last_ts = datetime.fromisoformat(last_ts_str)
            elapsed = (now_utc - last_ts).total_seconds()
            if elapsed < COOLDOWN_SECONDS and LEVEL_RANK.get(highest, 0) <= LEVEL_RANK.get(last_lvl, 0):
                rem_min = int((COOLDOWN_SECONDS - elapsed) // 60)
                return {
                    "sent": False,
                    "reason": f"Đang trong cooldown 3h cho mức {last_lvl} (còn {rem_min} phút).",
                    "level": highest,
                    "message": msg,
                }
        except Exception:
            pass

    if dry_run:
        return {"sent": False, "dry_run": True, "level": highest, "message": msg}

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return {
            "sent": False,
            "reason": "Thiếu TELEGRAM_BOT_TOKEN hoặc TELEGRAM_CHAT_ID trong .env / biến môi trường.",
            "level": highest,
            "message": msg,
        }

    resp = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
        timeout=15,
    )
    resp.raise_for_status()

    _save_state({
        "last_level": highest,
        "last_sent_utc": now_utc.isoformat(),
        "last_sent_vn": live.get("now_vn"),
    })
    return {"sent": True, "level": highest, "message": msg}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gửi cảnh báo Telegram EPB / RTK-GNSS.")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ chạy dự báo và in nội dung tin nhắn, không gửi.")
    parser.add_argument("--force", action="store_true", help="Bỏ qua cooldown 3 giờ.")
    args = parser.parse_args()

    out = send_telegram_alert(dry_run=args.dry_run, force=args.force)
    print(json.dumps(out, ensure_ascii=False, indent=2))
