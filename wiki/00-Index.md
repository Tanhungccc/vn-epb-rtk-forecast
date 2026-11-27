---
tags: [moc, index, vn-ionosphere]
updated: 2026-11-26
---

# Brain / Wiki — Dự báo Tầng điện ly & Nhiễu GNSS Việt Nam

> [!abstract] Mục tiêu cốt lõi
> Dự báo sớm **1–6 giờ** hiện tượng **Bong bóng Plasma Xích đạo (EPB)** và nhiễu/nhấp nháy tín hiệu GNSS (`ROTI`, $S_4$, `TEC`) trên lãnh thổ Việt Nam (`6°N–24°N, 101°E–116°E`, khung phân tích/hiển thị mở rộng `0°–28°N, 95°–122°E`), tập trung khung giờ tối **19h–24h LT (12–17 UTC)**, hoàn toàn từ dữ liệu mở.

## Bản đồ nội dung (MOC)

1. [[01-Muc-tieu-va-Lo-trinh]] — Bài toán ứng dụng VN, phạm vi địa lý, lộ trình train 2024–2025 / test mù 2026
2. **Nguồn dữ liệu mở & Bẫy hạ tầng**
   - [[Madrigal-TEC-ROTI]] — `kinst 8000` (`3500`, `3505`, `3506`, `3507`), giới hạn server >200 MB, độ trễ ≥5–7 ngày, độ phủ 2 dải vĩ độ
   - [[COSMIC-2-S4]] — `rapid/` vs `nrt/`, `scnLv2` (tầng F) vs `podTc2` vs `scn1c2`, hiện trạng 3/6 vệ tinh năm 2026
   - [[NOAA-SWPC-OMNI]] — `Kp`, `Bz` (`rtsw_mag_1m`), và các tiền tố cần bổ sung ($F_{10.7}$, $E_y$, `Dst/SYM-H`)
3. [[03-Vat-ly-va-Baseline]] — `ROTI-proxy` vs `ROTI` 30s, tiền tố dự báo 3–6h, tách baseline (`IRI-2020` vs `Persistence`/`Climatology`)
4. [[04-Nhat-ky-Thuc-nghiem]] — Số liệu chạy thật `2024-05-10`, `2024-05-11`, `2024-06-04`, `2026-11-20`
5. [[05-Quyet-dinh-Ky-thuat]] — Nhật ký các quyết định kiến trúc (ADR-001 → ADR-012)
6. [[06-Dashboard-va-Van-hanh]] — Thiết kế Dashboard 4 khung nhìn (2D + 3D lai + Keogram + Cảnh báo) & 2 luồng cập nhật dữ liệu
7. [[07-Mo-hinh-va-Hieu-chuan]] — `XGBoost-Cal` vs `LightGBM-Cal`, hiệu chuẩn xác suất, `TreeSHAP` & prospective 2026
8. [[09-De-cuong-Bai-bao]] — Đề cương bài báo / báo cáo khoa học hoàn chỉnh từ kết quả đã xác minh

## Codebase chính trong workspace
- `config.py` — Khung tọa độ VN, ngưỡng `ROTI_THRESHOLD = 0.5`, URL nguồn
- `src/fetch_madrigal.py` — Tải `kindat 3500` (TEC) & `3507` (ROTI)
- `src/roti.py` — Tính `ROT`, `ROTI-proxy` 5 phút, thống kê khung VN
- `src/roti_3507.py` — Đọc & cắt vùng `kindat 3507` theo chunk
- `src/scnlv2.py` — Tải & lọc `COSMIC-2 scnLv2_rapid` tầng F (`occheight >= 200 km`)
- `src/fetch_swpc.py` — Tải `Kp 1m` + `Bz rtsw_mag_1m`
- `src/validate_3way.py` — Kiểm chứng 3 chiều `scnLv2` vs `3507` vs `3500`
- `src/run_step1.py` — Pipeline 1 ngày (`--date YYYY-MM-DD`)
