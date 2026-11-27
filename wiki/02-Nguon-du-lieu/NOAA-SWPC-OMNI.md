---
tags: [data-source, swpc, solar-wind]
updated: 2026-11-26
---

# NOAA SWPC & Chỉ số Mặt Trời / Địa từ

← Quay lại [[00-Index]]

## 1. Đang dùng trong `src/fetch_swpc.py` (Zero-Auth JSON)
- **`Kp` 1 phút (Wing Kp):** `https://services.swpc.noaa.gov/json/planetary_k_index_1m.json`
- **`Bz` / `Bt` GSM realtime (1 phút):** `https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json`
  - *(Lưu ý: endpoint cũ của SWPC đã trả `404`, phải dùng `rtsw_mag_1m.json`.)*
- **Giới hạn hiện tại:** `fetch_swpc.py` mới lấy snapshot 1–7 ngày gần nhất, chưa kéo chuỗi lịch sử `2024–2025`.

## 2. Cần bổ sung cho tập dữ liệu lịch sử & dự báo 3–6h
Xem lý do vật lý tại [[03-Vat-ly-va-Baseline]]:
1. **$F_{10.7}$ (Solar Radio Flux 10.7 cm):** Kiểm soát mạnh nhất tần suất xuất hiện EPB theo chu kỳ Mặt Trời.
2. **Tốc độ gió Mặt Trời $v_{\text{sw}}$ & $B_z$ lịch sử (OMNIWeb / CDAWeb):** Tính điện trường liên hành tinh $E_y = -v_{\text{sw}} \cdot B_z$ (nhạy hơn $K_p$ 3 giờ để bắt điện trường xuyên thấu PPEF).
3. **Chỉ số `Dst` / `SYM-H`:** Định lượng pha chính / pha phục hồi của bão địa từ.
