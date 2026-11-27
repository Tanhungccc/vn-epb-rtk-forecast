---
tags: [paper, report, outline]
updated: 2026-11-27
---

# Đề cương Bài báo / Báo cáo Khoa học

← Quay lại [[00-Index]]

## Tên đề tài đề xuất
**Forecasting Equatorial Plasma Bubbles and GNSS Scintillation over Vietnam (1–6 h Lead) Using Multi-Source Open Observations and Calibrated Gradient-Boosted Residual Models**

## 1. Giới thiệu & Động lực (Introduction)
- Việt Nam (`6°–24°N, 101°–116°E`) nằm trải dài từ **Xích đạo từ (~9°N, Nam Bộ)** đến **đỉnh Dị thường Điện ly Xích đạo EIA (~20°N, Bắc Bộ)** ([[EIA]], [[EPB]]).
- Nhu cầu cảnh báo mất khóa pha **RTK-GNSS** và nhiễu định vị UAV/hàng hải mà không cần đầu tư mạng lưới phần cứng mới ([[01-Muc-tieu-va-Lo-trinh]]).

## 2. Dữ liệu & Kiểm chứng Chéo Đa nguồn (Data & Cross-Validation)
- **439 ngày (2024–2026):** `2024` (203d đủ 4 mùa), `2025` (115d Xuân+Thu), `2026` (121d kiểm thử mù) — tổng ~59.8 MB `.parquet`.
- **Phát hiện phương pháp luận ([[04-Nhat-ky-Thuc-nghiem]] Mốc 4):** Đối chứng 3 chiều chứng minh `ROTI-proxy` 5 phút (`Madrigal 3500`) chỉ đạt Spearman `0.176` so với [[ROTI]] chuẩn 30s từ [[IGS-30s-Stations]] → loại bỏ proxy, dùng trực tiếp **IGS 30s BKG** + **[[COSMIC-2-S4|`scnLv2_rapid`]]** (lọc tầng F `occheight >= 200 km`) + **[[ESA-GIM-IONEX]]** + **[[NOAA-SWPC-OMNI]]**.
- **Lưu ý 2026:** COSMIC-2 chỉ còn 3/6 vệ tinh (`C2E2, C2E5, C2E6`) → áp dụng **masked evaluation** ([[Prospective-Validation-2026]]).

## 3. Phương pháp Dự báo & Hiệu chuẩn (Methodology)
- Khung thời gian chống rò rỉ biên ngày (`groupby("date").shift(lead)` cho `lead = 1..6h`).
- So sánh `Persistence`, `Climatology`, `XGBoost-Cal`, `LightGBM-Cal`, và **`Climatology + XGBoost Residual` (`base_margin`)** ([[07-Mo-hinh-va-Hieu-chuan]]).
- Hiệu chuẩn xác suất `Sigmoid` & tối ưu ngưỡng `TSS` trên Validation.

## 4. Kết quả Thực nghiệm & Giải thích Vật lý (Results & Discussion)
- **Nowcast (`1h–2h`):** `LightGBM-Cal` / `XGBoost-Cal` đạt PR-AUC **`0.784` (`1h`)**, **`0.753` (`2h`)**, vượt `Climatology` (`0.713`) và `Persistence` (`0.644–0.659`) — dẫn chứng bằng `fig1_pr_curves_vn.png`, `fig2_calibration_reliability.png`.
- **Forecast (`3h–6h`):** `XGB-Resid` (`base_margin`) khắc phục trôi pha Mặt Trời 2024–2025 → 2026, đạt PR-AUC **`0.718–0.751`** (`vn 3–5h`) và **`0.686–0.755`** (`central 2–6h`).
- **[[TreeSHAP]] (`fig3_treeshap_1h_vs_6h.png`):** Chuyển dịch rõ từ IGS 30s ROTI (`1h`) sang tiền tố trước hoàng hôn (`6h`).
- **Case studies:** `fig4_2026_blind_cases.png`, `fig5_keograms_gannon.png`, và Dashboard [[06-Dashboard-va-Van-hanh]].

## 5. Hạn chế & Hướng phát triển (Limitations & Future Work)
- Bổ sung $dh'F/dt$ lúc chạng vạng (`17h–19h LT`) từ ionosonde `PHU` / `DLT` hoặc `COSMIC-2 ionPrf` để tăng độ chính xác tầm `3–6h`.
- Thay BKG archive bằng luồng **NTRIP RTCM 30s** để vận hành real-time từng phút.
