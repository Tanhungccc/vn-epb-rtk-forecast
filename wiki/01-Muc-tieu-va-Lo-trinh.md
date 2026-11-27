---
tags: [roadmap, goals, vietnam]
updated: 2026-11-26
---

# Mục tiêu & Lộ trình Dự án

← Quay lại [[00-Index]]

## 1. Vì sao bài toán có giá trị thực tiễn cao tại Việt Nam?
- **Cấu trúc Bắc–Nam đặc thù:** Lãnh thổ VN trải dài từ **xích đạo từ ở Nam Bộ (`~8°–10°N`, vùng sinh EPB)** lên tới **đỉnh Dị thường Điện ly Xích đạo (EIA) ở Bắc Bộ (`~18°–23°N`, vùng mật độ nền cao nhất gây $S_4$ cực đại)**.
- **Tác động trực tiếp:** EPB sau hoàng hôn (`19h–24h LT` = `12–17 UTC`) gây mất khóa pha trạm **RTK-GNSS** — ảnh hưởng khảo sát địa chính, nông nghiệp chính xác, UAV/drone, cảng biển và đồng bộ thời gian viễn thông.

## 2. Khung tham số chuẩn
- **Tọa độ:** `6°N – 24°N`, `101°E – 116°E`
- **Khung giờ EPB:** `12 – 17 UTC` (`19h – 24h` giờ Việt Nam)
- **Tầm dự báo:**
  - `1–2 giờ` (Nowcast sau 19h LT): dựa vào chuỗi `ROTI` / $S_4$ hiện tại + trôi Đông/lan Bắc.
  - `3–6 giờ` (Forecast từ 14h–17h LT): bắt buộc dùng [[03-Vat-ly-va-Baseline#Tiền tố vật lý cho tầm dự báo 3–6 giờ|tiền tố vật lý]] ($F_{10.7}$, $E_y$, `Dst`, độ chênh TEC Bắc–Nam, mùa).

## 3. Lộ trình thực hiện
1. **Bước 1 (Đã xong):** Pipeline `Madrigal 3500 + SWPC -> ROTI-proxy`, kiểm chứng trên `2024-05-10` vs `2024-06-04` ([[04-Nhat-ky-Thuc-nghiem]]).
2. **Bước 2 (Đang làm):**
   - Tích hợp [[COSMIC-2-S4|`COSMIC-2 scnLv2`]] (và/hoặc `podTc2`) làm phép đo $S_4$ trực tiếp.
   - Đối chứng `ROTI-proxy` (`3500`) vs `ROTI` chuẩn (`3507`) vs $S_4$ (`scnLv2`) trên tập ngày kiểm chứng.
   - Chốt schema `.parquet` gồm cả các biến tiền tố 3–6h.
3. **Bước 3 (Dataset nhiều năm & Mô hình):**
   - Kéo **400–600 ngày (2024–2025)** làm tập Train/Val theo **blocked time-series split** (chia theo khối ngày liên tục, không chia ngẫu nhiên theo giờ).
   - Giữ **năm 2026** làm **prospective out-of-time validation** (kiểm thử mù).
   - Chạy Baseline (`IRI-2020` cho TEC; `Persistence` + `Climatology` cho EPB) → `XGBoost`/`LightGBM` (+ SHAP) → `ConvLSTM`.
