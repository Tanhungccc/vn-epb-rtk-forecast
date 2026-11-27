# Dự báo Tầng điện ly Xích đạo & Nhiễu GNSS (EPB / S₄) tại Việt Nam

Hệ thống dự báo sớm **1–6 giờ** hiện tượng **Bong bóng Plasma Xích đạo (EPB)** và nhấp nháy tín hiệu GNSS (`S4 >= 0.50` tầng F, `occheight >= 200 km`) trên lãnh thổ Việt Nam (`6°–24°N, 101°–116°E`, khung khu vực `0°–28°N, 95°–122°E`), hoàn toàn từ dữ liệu mở.

## 1. Dữ liệu & Kết quả Kiểm thử mù 2026 (Prospective Validation)
- **Tập dữ liệu sạch (`data/dataset/`):** **439 ngày** (`2024`: 203 ngày, `2025`: 115 ngày → Train/Val; `2026`: 121 ngày → Blind Test).
  - `hourly_features.parquet` (**0.58 MB**) + `scnlv2_points.parquet` (**59.25 MB**).
- **Nguồn đầu vào:**
  - `COSMIC-2 scnLv2_rapid` (`occheight >= 200 km`): Nhãn thực đo $S_4$ tầng F (`vn`, `north`, `central`, `south`).
  - `IGS 30s` từ BKG (`HKSL`, `CUSV`, `PIMO`, `NTUS`): Tính ROTI chuẩn (Pi et al., 1997) cho nowcast `1–2h`.
  - `ESA GIM IONEX` + `NASA OMNI2`: Tiền tố trước hoàng hôn ($F_{10.7}$, $B_z$, $v_{\text{sw}}$, $E_y$, $K_p$, $Dst$, độ chênh EIA Bắc–Nam, $\sin/\cos\text{DOY}$).
- **Hiệu năng kiểm thử mù 2026 (`S4 >= 0.50`):**
  - `target=vn` (`1h–2h`): `LightGBM-Cal` / `XGBoost-Cal` đạt PR-AUC **`0.784` (`1h`)** và **`0.753` (`2h`)** (vượt `Climatology` `0.713` và `Persistence` `0.644–0.659`).
  - `target=vn` (`3h–5h`): Kỹ thuật lai `Climatology + XGBoost Residual` (`base_margin`) đạt PR-AUC **`0.718` (`3h`)**, **`0.751` (`4h`)**, **`0.726` (`5h`)** (vượt `Climatology` `0.713`).
  - `target=central` (`2h–6h`): `XGB-Resid` đạt PR-AUC **`0.686–0.755`** (vượt `Climatology` `0.673`).

## 2. Chạy lại toàn bộ từ `.parquet` (không cần tải lại dữ liệu thô)

```powershell
# 1. Cài môi trường
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt

# 2. Huấn luyện, hiệu chuẩn xác suất và đối chuẩn 6 tầm (1h..6h)
.\.venv\Scripts\python.exe -m src.train_models --targets vn north central south --s4-thresh 0.50

# 3. Xuất bộ biểu đồ chuẩn bài báo (300 DPI -> reports/figures/)
.\.venv\Scripts\python.exe -m src.export_figures

# 4. Mở Web App Dự báo Real-Time & Kiểm thử mù 2026
.\.venv\Scripts\streamlit.exe run app.py
```

## 3. Cấu trúc chính
```
src/
  batch_ingest.py      # Tải – Cắt – Nối – Xóa có checkpoint.json (2024–2026)
  igs_roti.py          # Tính ROTI chuẩn 30s (Pi et al., 1997) từ RINEX 3 (.crx.gz)
  scnlv2.py            # Đọc & lọc COSMIC-2 scnLv2_rapid tầng F (occheight >= 200 km)
  precursors.py        # OMNI2 + ESA GIM IONEX (tính EIA Bắc–Nam buổi chiều)
  residual_model.py    # Wrapper Climatology base_margin + XGBoost Residual
  train_models.py      # Huấn luyện, hiệu chuẩn Sigmoid/Isotonic, lưu models/*.joblib
  export_figures.py    # Xuất 5 biểu đồ 300 DPI vào reports/figures/
  app_demo.py          # Dashboard Streamlit (Dự báo 2026 + TreeSHAP + Case Study 3D)
wiki/                  # Obsidian Knowledge Vault (Graph View + Architecture.canvas)
```
