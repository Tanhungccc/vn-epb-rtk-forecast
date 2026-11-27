"""Telegram Chatbot tương tác cho Hệ thống Dự báo EPB & RTK-GNSS Việt Nam.

Các lệnh hỗ trợ trên Telegram:
  /start, /help  - Xem hướng dẫn & danh sách lệnh
  /dubao         - Xem dự báo rủi ro EPB 3 miền (Bắc / Trung / Nam) & khuyến nghị RTK/UAV
  /swpc          - Xem nhanh chỉ số gió Mặt Trời & bão từ thời gian thực (NOAA SWPC)
  /web           - Mở link Web App bản đồ 3D trực tuyến

Chạy bot (long-polling):
  python -m src.telegram_bot
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import live_forecast, send_alert  # noqa: E402

WEB_APP_URL = "https://vn-epb-rtk-forecast-57mcejdfsncm9vthegm3at.streamlit.app/"


def _fmt_num(val: float, fmt: str = ".1f", suffix: str = "") -> str:
    return f"{val:{fmt}}{suffix}" if isinstance(val, (int, float)) and np.isfinite(val) else "N/A"


def handle_command(cmd_text: str) -> str:
    """Xử lý lệnh chat từ người dùng và trả về nội dung Markdown."""
    cmd = cmd_text.strip().split()[0].split("@")[0].lower()

    if cmd in ("/start", "/help"):
        return (
            "🛰️ *Bot Dự báo Nhiễu Tầng điện ly EPB & RTK-GNSS Việt Nam*\n\n"
            "Các lệnh bạn có thể dùng:\n"
            "• `/dubao` — Dự báo xác suất EPB 3 miền (Bắc/Trung/Nam) & khuyến nghị RTK/UAV\n"
            "• `/swpc` — Chỉ số gió Mặt Trời & bão địa từ thời gian thực (NOAA SWPC)\n"
            "• `/web` — Link mở nhanh Web App 3D trực tuyến\n"
            "• `/help` — Hiện lại hướng dẫn này\n\n"
            f"🌐 Web App: {WEB_APP_URL}"
        )

    if cmd == "/web":
        return f"🌐 *Web App Dự báo EPB & Bản đồ 3D Việt Nam:*\n{WEB_APP_URL}"

    if cmd == "/swpc":
        live = live_forecast.run_live_forecast(save_log=False)
        sw = live["swpc"]
        return (
            f"☀️ *CHỈ SỐ THỜI TIẾT KHÔNG GIAN (NOAA SWPC)*\n"
            f"🕒 {live['now_vn']} | 📡 {live['freshness']['swpc']}\n\n"
            f"• Từ trường `Bz (GSM)`: *{_fmt_num(sw.get('bz_gsm'), '.2f', ' nT')}* (thấp nhất 6h: {_fmt_num(sw.get('bz_min_6h'), '.2f', ' nT')})\n"
            f"• Tốc độ gió Mặt Trời `v_sw`: *{_fmt_num(sw.get('v_sw'), '.0f', ' km/s')}*\n"
            f"• Điện trường `Ey`: *{_fmt_num(sw.get('ey_mvm'), '.2f', ' mV/m')}* (cao nhất 6h: {_fmt_num(sw.get('ey_max_6h'), '.2f', ' mV/m')})\n"
            f"• Chỉ số bão từ `Kp`: *{_fmt_num(sw.get('kp'), '.1f')}* | `Dst`: *{_fmt_num(sw.get('dst'), '.0f', ' nT')}*\n"
            f"• Bức xạ Mặt Trời `F10.7`: *{_fmt_num(sw.get('f107'), '.0f', ' sfu')}*"
        )

    if cmd in ("/dubao", "/forecast", "/status"):
        live = live_forecast.run_live_forecast()
        fc = live["forecast_df"]
        if fc.empty:
            return "⚠️ Hiện chưa có dữ liệu mô hình để dự báo."

        lines = [
            f"📡 *DỰ BÁO EPB & RỦI RO RTK-GNSS / UAV*",
            f"🕒 {live['now_vn']} | {live['mode_label']}",
            f"🎯 Độ tin cậy dữ liệu: *{live['data_confidence']}%*",
            "",
        ]
        for tgt, name in [
            ("north", "Bắc Bộ (18°–24°N)"),
            ("central", "Trung Bộ (12°–18°N)"),
            ("south", "Nam Bộ (6°–12°N)"),
        ]:
            sub = fc[fc["target"] == tgt]
            if sub.empty:
                continue
            idx_max = sub["prob"].idxmax()
            p_max = float(sub.loc[idx_max, "prob"])
            peak_lt = str(sub.loc[idx_max, "lt_label"])
            if p_max >= 0.65:
                icon, lvl = "🔴", "ĐỎ"
            elif p_max >= 0.35:
                icon, lvl = "🟡", "VÀNG"
            else:
                icon, lvl = "🟢", "XANH"
            lines.append(f"{icon} *{name}*: {lvl} (`{p_max*100:.1f}%` lúc `{peak_lt}`)")

        vn_sub = fc[fc["target"] == "vn"]
        vn_max = float(vn_sub["prob"].max()) if not vn_sub.empty else 0.0
        if vn_max >= 0.65:
            advice = "Tạm dừng đo RTK độ chính xác cao & hạn chế bay UAV buổi tối."
        elif vn_max >= 0.35:
            advice = "Thời gian Fix RTK có thể chậm; hạn chế bay UAV ngoài tầm nhìn (BVLOS)."
        else:
            advice = "Điều kiện tầng điện ly ổn định, hoạt động RTK/UAV bình thường."

        lines.extend(["", f"👉 *Khuyến nghị:* {advice}", f"🌐 Chi tiết 3D: {WEB_APP_URL}"])
        return "\n".join(lines)

    return "Lệnh chưa hỗ trợ. Gửi `/help` để xem danh sách lệnh (`/dubao`, `/swpc`, `/web`)."


def _send_reply(token: str, chat_id: int | str, text: str) -> None:
    requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown", "disable_web_page_preview": True},
        timeout=15,
    )


def register_commands(token: str) -> dict:
    """Đăng ký menu lệnh `/` hiển thị sẵn trên khung chat Telegram."""
    commands = [
        {"command": "dubao", "description": "Dự báo EPB 3 miền & khuyến nghị RTK/UAV"},
        {"command": "swpc", "description": "Chỉ số gió Mặt Trời & bão từ (NOAA SWPC)"},
        {"command": "web", "description": "Mở Web App bản đồ 3D trực tuyến"},
        {"command": "help", "description": "Xem hướng dẫn sử dụng bot"},
    ]
    r = requests.post(
        f"https://api.telegram.org/bot{token}/setMyCommands",
        json={"commands": commands},
        timeout=15,
    )
    return r.json()


def run_polling() -> None:
    send_alert.load_env_file()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Thiếu TELEGRAM_BOT_TOKEN trong .env hoặc biến môi trường.")

    reg = register_commands(token)
    print(f"Bot đang chạy (setMyCommands={reg.get('ok')})... Nhấn Ctrl+C để dừng.", flush=True)

    offset = None
    while True:
        try:
            params = {"timeout": 25}
            if offset is not None:
                params["offset"] = offset
            resp = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", params=params, timeout=35).json()
            for upd in resp.get("result", []):
                offset = upd["update_id"] + 1
                msg = upd.get("message") or {}
                text = (msg.get("text") or "").strip()
                chat_id = (msg.get("chat") or {}).get("id")
                if chat_id and text.startswith("/"):
                    reply = handle_command(text)
                    _send_reply(token, chat_id, reply)
                    print(f"[{chat_id}] {text} -> replied", flush=True)
        except KeyboardInterrupt:
            print("Đã dừng bot.", flush=True)
            break
        except Exception as e:  # noqa: BLE001
            print(f"Lỗi polling: {e}", flush=True)
            time.sleep(5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Telegram Chatbot EPB / RTK-GNSS.")
    ap.add_argument("--test-cmd", default="", help="Chạy thử một lệnh (vd: /dubao) mà không mở polling.")
    args = ap.parse_args()
    if args.test_cmd:
        print(handle_command(args.test_cmd))
    else:
        run_polling()
