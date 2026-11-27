---
tags: [ml, calibration, shap, forecasting]
updated: 2026-11-26
---

# Mô hình Dự báo, Hiệu chuẩn Xác suất & Giải thích (TreeSHAP)

← Quay lại [[00-Index]] | Xem kết quả: [[04-Nhat-ky-Thuc-nghiem]]

## 1. Kiến trúc Huấn luyện (`src/train_models.py`)
- **Mục tiêu chính:** `target=vn` (`6°–24°N, 101°–116°E`) và 3 miền (`north` 18–24°N, `central` 12–18°N, `south` 6–12°N) ở ngưỡng nhấp nháy mạnh **`S4 >= 0.50`** trên [[COSMIC-2-S4|`COSMIC-2 scnLv2`]] (lọc tầng F `occheight >= 200 km`).
  *(Tránh dùng `regional S4 >= 0.30` làm thước đo chính vì tỷ lệ dương tới ~90% làm TSS mất ổn định.)*
- **Chống rò rỉ dữ liệu:**
  - Mọi phép `.shift(lead)` cho `lead = 1..6h` đều nằm trong `groupby("date")`.
  - **Prospective Validation:** Train/Val trên **2024–2025** (phá bẫy `cos_doy` khi chỉ có 1 năm 2024), kiểm thử mù hoàn toàn trên **2026** (có mask theo các ô/giờ `scnLv2` đi qua).

## 2. Mô hình & Hiệu chuẩn
- **Baselines ([[03-Vat-ly-va-Baseline]]):** `Persistence` và `Climatology` (`utc_hour` + `F10.7 >= 150`).
- **ML Tabular:** `XGBoost` vs `LightGBM` với `scale_pos_weight`.
- **Probability Calibration:** Dùng `CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid"|"isotonic")` fit trên tập Validation (20% cuối của khối Train), sau đó tối ưu ngưỡng **TSS** trên Validation để xuất Brier Score & xác suất cảnh báo chuẩn cho [[06-Dashboard-va-Van-hanh]].
- **Lưu trữ:** `models/{target}_lead{h}h.joblib` + `models/benchmark_summary.json`.

## 3. Giải thích Vật lý (`TreeSHAP`)
- Dùng `pred_contribs=True` tích hợp sẵn của XGBoost/LightGBM để vẽ biểu đồ đóng góp log-odds trực tiếp trên [[06-Dashboard-va-Van-hanh|Dashboard]].
- Cơ chế đã kiểm chứng:
  - `lead = 1h`: [[Madrigal-TEC-ROTI|IGS 30s ROTI]] (`HKSL`, `CUSV`, `PIMO`) dẫn đầu.
  - `lead = 6h`: Nhóm tiền tố trước hoàng hôn ([[NOAA-SWPC-OMNI|`v_sw`, `Dst`, `Kp`, `F10.7`, `Bz`]] + `ESA IONEX EIA Bắc/Nam` + `sin/cos DOY`) chiếm ưu thế.
