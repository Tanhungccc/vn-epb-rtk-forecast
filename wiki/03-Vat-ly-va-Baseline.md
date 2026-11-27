---
tags: [physics, baseline, features]
updated: 2026-11-26
---

# Vật lý Điện ly, Tiền tố Dự báo & Baseline

← Quay lại [[00-Index]]

## 1. `ROTI-proxy` (5 phút) vs `ROTI` chuẩn (30 giây)
- **Hiện tại (`src/roti.py`):** Tính từ lưới `kindat 3500` (1°×1°×5 phút):
  `ROT = (TEC_t - TEC_{t-5min}) / 5`, `ROTI = std(ROT)` trên 5 map (20 phút).
- **Chuẩn vật lý (Pi et al., 1997):** Tính từ `ΔSTEC / Δt` bước **30 giây** dọc cung LOS vệ tinh–trạm thu trong cửa sổ 5 phút.
- **Hướng kiểm chứng:** Đối chiếu `ROTI-proxy` (`3500`) với `kindat 3507` (ROTI tính sẵn từ LOS 30s) và `COSMIC-2 scnLv2` ($S_4$) trên các ngày bão/tĩnh mẫu.

## 2. Tách Baseline theo đúng đại lượng dự báo
- **Dự báo bản đồ nền `TEC`:** So với **`IRI-2020`**; dùng chính $\Delta\text{TEC} = \text{TEC}_{\text{obs}} - \text{TEC}_{\text{IRI}}$ làm feature đầu vào cho ML.
- **Dự báo `EPB` / `ROTI` / $S_4$:** `IRI-2020` không xuất ra bất đồng nhất/nhấp nháy, nên dùng 2 baseline chuẩn:
  1. **Persistence baseline:** trạng thái tại $t + \Delta t$ bằng trạng thái tại $t$.
  2. **Climatology baseline:** xác suất lịch sử theo tháng + giờ địa phương + mức $F_{10.7}$.

## 3. Tiền tố vật lý cho tầm dự báo 3–6 giờ
Trước `18:30 LT`, `ROTI` và $S_4$ gần như bằng `0`, nên để dự báo từ `14h–17h LT` cho buổi tối cần thêm:
1. **$F_{10.7}$** (thông lượng vô tuyến Mặt Trời 10.7 cm).
2. **$E_y = -v_{\text{sw}} \cdot B_z$** và **`Dst` / `SYM-H`** (từ OMNIWeb / SWPC) để bắt hiện tượng PPEF lúc chạng vạng.
3. **Độ chênh / tỷ số `TEC` Bắc–Nam lúc `14h–17h LT`** (tính trực tiếp từ lưới Madrigal giữa Bắc Bộ `19°–23°N` và Nam Bộ `8°–11°N` — proxy cho hiệu ứng đài phun nước xích đạo).
4. **Tính tuần hoàn mùa:** $\sin(\text{DOY}), \cos(\text{DOY})$ bắt 2 đỉnh phân điểm (tháng 3–4, 9–10).
5. *(Mở rộng nếu có)* $dh'F/dt$ lúc `17h–19h LT` từ ionosonde `PHU` / `DLT`.
