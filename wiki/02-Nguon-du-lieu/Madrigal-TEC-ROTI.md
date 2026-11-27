---
tags: [data-source, madrigal, tec, roti]
updated: 2026-11-26
---

# CEDAR Madrigal GNSS-TEC & ROTI (`kinst = 8000`)

← Quay lại [[00-Index]]

## 1. Các `kindat` hiện có (kiểm tra ngày `2024-05-10`)

| `kindat` | Tên sản phẩm | File mẫu (`2024-05-10`) | Dung lượng / ngày | Ghi chú sử dụng |
| --- | --- | --- | --- | --- |
| `3500` | *TEC binned 1° × 1° × 5 min* | `gps240510g.003.hdf5` | **~150 MB** | Đang dùng trong `src/fetch_madrigal.py` |
| `3505` | *Line of sight TEC data (30s)* | `los_20240510.002.h5` | **13,993.62 MB (~14.0 GB)** | Quá nặng để kéo hàng loạt |
| `3506` | *List of sites used in daily TEC* | `site_20240510.002.h5` | **0.13 MB** | Dùng tra tọa độ trạm thực tế |
| `3507` | *Calculated ROTI (`TECu/sec`)* | `roti_131_2024.h5` | **656.87 MB** | Đã tính sẵn ROTI từ LOS 30s — hợp để đối chứng |

## 2. Giới hạn hạ tầng & Bẫy đã xác minh
- **Server không hỗ trợ HTTP Range:** `src/fetch_madrigal.py` tự xóa file lỗi và tải lại từ đầu.
- **Không lọc phía server cho file > 200 MB:** Thử `isprintService.py` với filter `gdlat 6..24, glon 101..116, hour 12..17` trên `roti_131_2024.h5` (656.87 MB) bị từ chối:
  > `File .../roti_131_2024.h5 greater than 200 MB in size - running dynamic file creation not possible. Please use -- download as is -- instead.`
  $\rightarrow$ Bắt buộc tải nguyên file về máy rồi mới lọc.

## 3. Độ phủ trạm & ô lưới tại Việt Nam
- Trong `site_20240510.h5` (`3506`): toàn cầu có `5,648` trạm; riêng hộp `6–24°N, 101–116°E` có **17 trạm**, nhưng **100% nằm ở Hồng Kông / Ma Cao / Hải Nam / Quảng Tây (`19.21°N – 23.50°N`)** — không có trạm nào trong đất liền VN từ `18°N` trở xuống.
- Trên lưới `kindat 3500` khung VN (`19 × 16 × 72 = 21,888` ô-slot trong `12–17 UTC`):
  - Ngày `2024-05-10`: **29.84%** (`6,532 / 21,888`) ô non-NaN.
  - Ngày `2024-06-04`: **17.12%** (`3,748 / 21,888`) ô non-NaN.
