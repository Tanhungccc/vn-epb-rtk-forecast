---
tags: [data-source, cosmic2, s4, scintillation]
updated: 2026-11-26
---

# UCAR COSMIC-2 Scintillation ($S_4$) & TEC

← Quay lại [[00-Index]]

## 1. Cấu trúc nhánh trên `data.cosmic.ucar.edu/gnss-ro/cosmic2/`
Kiểm tra trực tiếp ngày `2024-05-10` (DOY `131`), `2024-05-11` (DOY `132`), và `2026-11-20` (DOY `324`):
- `postProc/level1b`, `postProc/level2`: **HTTP 404**.
- `provisional/spaceWeather/`: `level1b` **HTTP 404**; `level2` chỉ có `ionPrf_prov1`.
- `nrt/level1b/`: có `podTc2_nrt_YYYY_DDD.tar.gz` (`951.3 MB` cho 2024/131; `543.3 MB` cho 2026/324).
- **`rapid/` (chứa đủ cả 3 sản phẩm cho cả 2024 lẫn 2026):**
  - `rapid/level1b/YYYY/DDD/podTc2_rapid_YYYY_DDD.tar.gz` (`764.0 MB` 2024/131; `390.4 MB` 2026/324)
  - `rapid/level1b/YYYY/DDD/scn1c2_rapid_YYYY_DDD.tar.gz` (`87.68 MB` 2024/131; `44.81 MB` 2026/324)
  - `rapid/level2/YYYY/DDD/scnLv2_rapid_YYYY_DDD.tar.gz` (`39.81 MB` 2024/131; `47.57 MB` 2024/132; `6.34 MB` 2026/324)

> [!note] Lưu ý đặt tên & tốc độ tải
> - Sản phẩm level-2 scintillation trên UCAR tên là **`scnLv2`** (không phải `scnLv1`).
> - Tốc độ tải thực tế từ UCAR đo được ~`170–250 KB/s`; mirror UMD (`gpsmet.umd.edu`) không có `podTc2`/`scnLv2`.

## 2. So sánh `scnLv2` vs `podTc2` vs `scn1c2`

| Sản phẩm | Vùng độ cao đo | Tọa độ sẵn có | Biến $S_4$ / RFI / Quality | Đánh giá cho bài toán VN |
| --- | --- | --- | --- | --- |
| **`scnLv2`** (`rapid/level2`) | **Xuyên tầng F (`occheight` 63.1–534.2 km, median 406.3 km)** | Có sẵn `lat`, `lon`, `occheight`, `elevation` | `s4_L1`, `s4_L2`, `sigma_phi_L1/L2`, `slip_L1/L2`, `RFI`, `L1_SNR`, `L2_SNR`, attrs `bitstatus1/2` | **Ưu tiên số 1:** nhẹ (6–48 MB/ngày), đo đúng tầng F, có **27 lát cắt / 3,710 điểm** trong đêm VN `2024-05-10` |
| **`podTc2`** (`rapid/level1b` / `nrt`) | Quỹ đạo POD (~520–550 km, `occheight` ~405–535 km khi âm) | Chỉ có `x_LEO/y_LEO/z_LEO` (ECF km) + attrs `lat_start/stop` | `S4`, `TEC`, `RFI`, `Aero_RFI`, `absolute_tec_indicator`, `attflag`, `podflag`, `ion_occ_score` | Dày hơn nhưng nặng (390–950 MB/ngày) và bỏ sót EPB thấp chưa dâng lên >520 km |
| **`scn1c2`** (`rapid/level1b`) | Quỹ đạo LEO (~532–538 km) | ECF `x_LEO/y_LEO/z_LEO` | `s4_L1`, `s4_L2`, `sigma_phi_L1/L2`, `RFI`, `Aero_RFI` | Trung gian (45–88 MB/ngày) |
