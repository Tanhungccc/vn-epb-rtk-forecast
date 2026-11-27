---
tags: [dashboard, visualization, operations, pipeline]
updated: 2026-11-26
---

# Thiết kế Dashboard & Kiến trúc Cập nhật Dữ liệu

← Quay lại [[00-Index]]

## Phần 1: Dashboard Trực quan hóa 4 Khung nhìn (`Streamlit + Plotly / PyDeck`)

Khung hiển thị khu vực mở rộng **`0°–28°N, 95°–122°E`** (vẽ nổi ranh giới vùng chấm điểm Việt Nam **`6°–24°N, 101°–116°E`**).

### 1. Bản đồ 2D Khu vực + Time Slider (`+1h .. +6h`)
5 lớp chồng nhau + chế độ so sánh song song **Predicted vs. Ground Truth**:
- **Lớp nền (Heatmap):** Xác suất EPB (`0–100%`) hoặc $\Delta\text{TEC} = \text{TEC}_{\text{obs}} - \text{TEC}_{\text{IRI-2020}}$. **Bắt buộc có mask/gạch chéo cho ô `NaN`** (đặc biệt dải `6°–18°N` nơi độ phủ `3500` chỉ ~13–17%) để phân biệt vùng yên tĩnh thật với vùng thiếu trạm.
- **Đường tham chiếu vật lý:** Đường **Xích đạo từ (Dip Equator, ~9°N)** và **đỉnh EIA phía Bắc (~19°–21°N)**.
- **Lớp điểm đo mặt đất (`Madrigal 3507` / IGS):** Các trạm quanh biên (Hồng Kông, Hải Nam, `CUSV` Bangkok, `PIMO` Manila, `NTUS` Singapore) tô màu theo ROTI 30s.
- **Lớp vệt cắt vệ tinh (`COSMIC-2 scnLv2`):** Hình chiếu mặt đất tô màu `s4_L1`: Xanh (`< 0.2`), Vàng (`0.2–0.5`), Đỏ (`> 0.5`).

### 2. Bản đồ 3D "Lai" (3D Hybrid View — Demo / Thuyết trình)
- **Mặt đáy (`z = 0 km`):** Bản đồ 2D khu vực + vùng cảnh báo rủi ro EPB tại VN.
- **Tầng điện ly (`z = 200–534 km`):** Các cung lát cắt `COSMIC-2 scnLv2` theo đúng `(lon, lat, occheight_km)`.
- **Toggle tách lớp độ cao:** Nút bật/tắt tách riêng **tầng F (`occheight >= 200 km`)** khỏi **vùng thấp `< 200 km` (Sporadic-E)** để chứng minh mô hình bắt đúng lõi tầng F (300–450 km).

### 3. Cặp biểu đồ Keogram (Cho báo cáo tĩnh / bài báo PDF)
- **Keogram Kinh độ – Thời gian (`95°E → 122°E` theo `19h → 24h LT`):** Đo vận tốc trôi Đông của EPB (~100–150 m/s) từ độ dốc vệt đỏ.
- **Keogram Vĩ độ – Thời gian (`0°N → 28°N` theo `19h → 24h LT`):** Cho thấy plasma bùng phát từ Dip Equator (~9°N) và lan lên vĩ độ cao (22°–24°N).

### 4. Bảng điều khiển Tiền tố & Cảnh báo 3 miền
- **4 chuỗi đầu vào đồng bộ:** $B_z$, $E_y = -v_{\text{sw}} \cdot B_z$, $Dst/K_p$, và độ chênh TEC Bắc–Nam buổi chiều (`14h–17h LT`).
- **Bảng trạng thái 3 miền (Bắc Bộ / Trung Bộ / Nam Bộ):** Quy đổi xác suất EPB / $S_4$ sang 3 cấp rủi ro RTK-GNSS & UAV (Xanh / Vàng / Đỏ).

---

## Phần 2: Hai Luồng Cập nhật Dữ liệu (Đã hiệu chỉnh theo hạ tầng thực tế)

### 1. Luồng Nghiên cứu / Kiểm thử mù (Historical + Replay 2026)
- **Batch Ingestion (2024–2025, ~400–600 ngày):** Kéo **`Madrigal 3500` (~150 MB/ngày)** + **`COSMIC-2 scnLv2_rapid` (~35–45 MB/ngày)** + **`NOAA SWPC / OMNI`** ($K_p, B_z, v_{\text{sw}}, F_{10.7}, Dst$).
- **Madrigal `3507` (657 MB/ngày):** **Chỉ tải ~20–30 ngày mẫu (bão + tĩnh)** để đối chứng và hiệu chỉnh `ROTI-proxy`, không tải cho cả 500 ngày vì server tốc độ ~230 KB/s, cấm `Range` và dễ reset sau ~1 giờ.
- **Cơ chế "Tải – Cắt – Nối – Xóa":** Cắt theo khung `0°–28°N, 95°–122°E`, lọc `occheight >= 200 km`, lưu `.parquet` phân vùng theo tháng (`year=YYYY/month=MM/`), xóa file thô để tổng dung lượng kho sạch gọn trong vài trăm MB.
- **Lưu ý kiểm thử mù 2026:**
  - Madrigal `kinst 8000` thực tế trễ **≥ 5–7 ngày** (ngày `2026-11-20` vẫn chưa có trên server vào `2026-11-26`), nên kiểm thử 2026 chạy theo dạng **replay/hindcast** ngay khi dữ liệu lên kệ.
  - `COSMIC-2` cuối năm 2026 chỉ còn **3/6 vệ tinh (`C2E2, C2E5, C2E6`)** $\rightarrow$ đánh giá `scnLv2` trên khung rộng `0°–28°N, 95°–122°E` và tổng hợp theo cửa sổ **tuần/tháng** thay vì từng ngày đơn lẻ.

### 2. Luồng Vận hành Real-Time (Nếu triển khai cảnh báo tức thời 14h–18h30 LT)
- **Nhịp 15 phút (SWPC):** `B_z`, `v_sw` (trễ 1–5 phút), `K_p` (3h), `F_10.7` (1 lần/ngày lúc 20:00 UTC).
- **Đầu vào GNSS buổi chiều (`14h–18h30 LT`):** Do Madrigal trễ ≥5–7 ngày, khi vận hành real-time thật phải thay Madrigal bằng luồng **NTRIP RTCM 30s** từ các trạm mở quanh biên (`CUSV`, `HKSL`, `PIMO`, `NTUS`) — mô hình ML phía sau giữ nguyên.
