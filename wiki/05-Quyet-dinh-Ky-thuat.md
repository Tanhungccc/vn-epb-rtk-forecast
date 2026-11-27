---
tags: [adr, decisions]
updated: 2026-11-26
---

# Nhật ký Quyết định Kỹ thuật (ADR)

← Quay lại [[00-Index]]

- **ADR-001 (Bước 1):** Dùng `Madrigal kindat 3500` (~150 MB/ngày) làm xương sống ban đầu + `SWPC` (`kp_1m`, `rtsw_mag_1m`), chưa gộp `INTERMAGNET PHU/DLT`.
- **ADR-002 (Cặp test đầu):** Dùng `2024-05-10` (bão Gannon G5) vs `2024-06-04` (ngày tĩnh) thay vì 7 ngày NRT cuối tháng 11/2026 vì NRT hiện tại $K_p \approx 1$.
- **ADR-003 (Bỏ `Madrigal 3505` và `3507`, dùng IGS 30s để đối chứng):** `3505` nặng `~14 GB/ngày`; `3507` nặng `657 MB/ngày` và server Madrigal thường xuyên ngắt kết nối (`curl exit 56`, cấm `Range`) nên **dứt khoát bỏ `3507`**. Thay bằng kéo trực tiếp **RINEX 30s (`.crx.gz`, ~2–5 MB/trạm/ngày)** của 4 trạm IGS quanh biên (`HKSL`, `CUSV`, `PIMO`, `NTUS`) từ mirror công khai không cần đăng nhập (BKG/IGN) để tính ROTI chuẩn (Pi et al., 1997) đối chứng với `ROTI-proxy` (`3500`).
- **ADR-004 (Chọn `COSMIC-2 scnLv2` nhánh `rapid/` + đánh giá có mask năm 2026):** Nhẹ (`6–48 MB/ngày`), đo thẳng qua tầng F (`occheight >= 200 km`), có sẵn `lat/lon` và `s4_L1/L2`. Do cuối 2026 chỉ còn 3/6 vệ tinh (`C2E2, C2E5, C2E6`), mọi phép tính metric (F1, Recall, RMSE) trên tập 2026 **bắt buộc mask theo đúng các ô/thời điểm có quỹ đạo `scnLv2` đi qua**, tuyệt đối không coi vùng trống vệ tinh là "không có EPB".
- **ADR-005 (Chiến lược ML & Split):** Train/Val trên `2024–2025` (400–600 ngày) bằng **blocked time-series split**; giữ **năm 2026** làm kiểm thử mù dạng replay/hindcast (do Madrigal trễ ≥5–7 ngày). Chạy `XGBoost`/`LightGBM` trước khi thử `ConvLSTM`.
- **ADR-006 (Obsidian Vault):** Đặt Vault tại `d:\VN research\wiki\` (tách khỏi `.venv/` và `data/`).
- **ADR-007 (Khung không gian & Trực quan hóa):** Mở rộng khung phân tích/hiển thị ra `0°–28°N, 95°–122°E` (highlight VN `6°–24°N, 101°–116°E`), dùng **2D Time Slider + mask vùng NaN** kết hợp **3D Hybrid (`occheight_km` thật, toggle tầng F ≥200 km vs Sporadic-E <200 km)**, đường **Dip Equator**, và cặp **Keogram (Lon–Time, Lat–Time)**.
- **ADR-008 (Tải Madrigal `3500` đơn luồng):** Server Madrigal tốc độ ~230 KB/s, cấm `Range` và dễ reset kết nối; tải tuần tự từng file `3500` (~150 MB/ngày) với `curl -4` + kiểm tra toàn vẹn HDF5.
- **ADR-009 (Dùng ESA GIM IONEX cho tiền tố EIA buổi chiều):** Vì `Madrigal 3500` trống 70–83% ô lưới trên VN, dùng bản đồ toàn cầu **ESA GIM IONEX** (`navigation-office.esa.int`, 1h rapid ~1 MB/ngày, zero-auth) để tính độ chênh TEC Bắc–Nam lúc `14h–17h LT`.
- **ADR-010 (Không dùng `3500 ROTI-proxy` làm nhãn hay đặc trưng ROTI chính):** Kết quả đối chứng cho thấy `3500 ROTI-proxy` chỉ đạt Spearman `0.176` (5-min) và `-0.015` (hourly) so với `IGS 30s ROTI`. Do đó giữ `3500` cho bản đồ nền `TEC` / $\Delta\text{TEC}$, còn `ROTI` mặt đất lấy từ **IGS 30s BKG** (`~15 MB/ngày` cho 4 trạm) kết hợp **$S_4$ của `COSMIC-2 scnLv2`**.
- **ADR-011 (Tinh gọn Batch 2024–2026: thay `Madrigal 3500` bằng `ESA GIM IONEX`):** Vì `3500` nặng ~150 MB/ngày nhưng trống 70–83% ô trên VN và server Madrigal chậm/chập chờn, giai đoạn tải hàng loạt 400–600 ngày dùng **`ESA GIM IONEX` (~1 MB/ngày, phủ kín 100%) + `IGS 4 trạm 30s` (~15 MB/ngày) + `COSMIC-2 scnLv2` (~6–48 MB/ngày) + `OMNI2` (~3.5 MB/năm)** → tổng dung lượng thô giảm còn ~35–60 MB/ngày.
- **ADR-012 (Chuyển trọng tâm `target=vn`/3 miền ở `S4 >= 0.50` & kéo đa năm 2024–2026):** `regional S4 >= 0.30` bị bẫy tỷ lệ dương ~90%, và train/test trong 1 năm 2024 khiến `cos_doy` học vẹt ranh giới mùa. Xem chi tiết tại [[07-Mo-hinh-va-Hieu-chuan]].
