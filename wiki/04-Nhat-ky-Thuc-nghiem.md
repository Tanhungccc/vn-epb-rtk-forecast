---
tags: [experiments, logs, validation]
updated: 2026-11-26
---

# Nhật ký Thực nghiệm

← Quay lại [[00-Index]]

## Mốc 1: Kiểm chứng Bước 1 (`Madrigal 3500` + `SWPC` → `ROTI-proxy`)
So sánh cặp đối chứng **Bão Gannon 10/05/2024 (G5)** vs **Ngày tĩnh 04/06/2024** trong khung tối VN (`12–17 UTC` = `19–24h LT`):

| Chỉ số | Bão `2024-05-10` | Tĩnh `2024-06-04` |
| --- | --- | --- |
| Ô `EPB-flag` (`ROTI ≥ 0.5 TECU/phút`) | **2 363** | **695** |
| `ROTI` max | **6.72 TECU/phút** | **4.34 TECU/phút** |
| Tỉ lệ non-NaN khung VN (`21,888` ô-slot) | **29.84%** (`6,532`) | **17.12%** (`3,748`) |

→ Tỉ lệ ô `EPB-flag` giữa ngày bão và ngày tĩnh đạt **~3.4×**.

## Mốc 2: Phân tích 4 ngày `COSMIC-2 scnLv2_rapid` (lọc `occheight >= 200 km`, `bitstatus == 0`, `RFI <= 0.05`)

| Ngày (DOY) | Dung lượng | File toàn cầu (`bitstatus_ok`) | Vệ tinh hoạt động | VN cả ngày (`raw` → `F≥200km`) | VN tối `12–17 UTC` | `s4_L1` tối VN |
| --- | --- | --- | --- | --- | --- | --- |
| **2024-05-10** (131) | 39.81 MB | 881 (`527`) | **6/6** (`C2E1..C2E6`) | 25f/3,378p → **21f/2,711p** | **14f / 2,039p** | max **0.831**, p95 **0.622**, `≥0.3`: **668** |
| **2024-05-11** (132) | 47.57 MB | 1,005 (`644`) | **6/6** (`C2E1..C2E6`) | 22f/3,040p → **15f/2,047p** | **4f / 456p** | max **0.613**, p95 **0.478**, `≥0.3`: **125** |
| **2024-06-04** (156) | 34.25 MB | 732 (`469`) | **6/6** (`C2E1..C2E6`) | 4f/628p → **4f/488p** | **3f / 315p** | max **0.333**, p95 **0.293**, `≥0.3`: **13** |
| **2026-11-20** (324) | **6.34 MB** | 254 (`124`) | **3/6** (`C2E2, C2E5, C2E6`) | 0f/0p → **0f/0p** | **0f / 0p** | — |

**Kết luận nguyên nhân `2026-11-20` chỉ 6.34 MB:** thiếu 3 vệ tinh (`C2E1, C2E3, C2E4`) và độ dài trung bình mỗi lát cắt giảm gần một nửa (`209.4` điểm/file vs `405.3` điểm/file ngày `2024-06-04`).

## Mốc 3: Độ phủ 2 dải vĩ độ `Madrigal 3500` (`12–17 UTC`)

| Ngày | Khung kinh độ | Dải Bắc (`18°–24°N`) | Dải Trung–Nam (`6°–18°N`) |
| --- | --- | --- | --- |
| **2024-05-10** | `101°–116°E` | **51.14%** (`4,124/8,064`) | **17.42%** (`2,408/13,824`) |
| **2024-05-10** | `99°–122°E` | **48.84%** (`5,908/12,096`) | **17.72%** (`3,675/20,736`) |
| **2024-05-11** | `101°–116°E` | **49.49%** (`3,991/8,064`) | **16.83%** (`2,327/13,824`) |
| **2024-05-11** | `99°–122°E` | **48.71%** (`5,892/12,096`) | **17.76%** (`3,682/20,736`) |
| **2024-06-04** | `101°–116°E` | **24.36%** (`1,964/8,064`) | **12.91%** (`1,784/13,824`) |
| **2024-06-04** | `99°–122°E` | **29.27%** (`3,541/12,096`) | **14.61%** (`3,029/20,736`) |

→ Mở rộng biên sang `99°–122°E` tăng thêm **~50% số ô có dữ liệu** ở dải Trung–Nam (`2,408 → 3,675` ô).

## Mốc 4: Đối chứng 3 chiều `IGS 30s ROTI` (Pi et al., 1997) vs `Madrigal 3500 ROTI-proxy` vs `COSMIC-2 scnLv2`
4 trạm IGS (`CUSV`, `HKSL`, `NTUS`, `PIMO`) tải từ BKG (`3.2–4.9 MB/trạm/ngày`), khung `12–17 UTC` qua 3 ngày `2024-05-10`, `2024-05-11`, `2024-06-04`:

- **IGS 30s `ROTI p95` phân tách rõ bão vs tĩnh:**
  - `2024-05-10` (Gannon): `CUSV=0.861`, `HKSL=0.867`, `NTUS=1.188`, `PIMO=1.290` (`578–1,171` mẫu `≥0.5`/trạm)
  - `2024-05-11`: `0.167–0.259` (`24–120` mẫu `≥0.5`)
  - `2024-06-04` (tĩnh): `0.128–0.250` (`8–164` mẫu `≥0.5`)
- **Tương quan `IGS 30s ROTI` vs `Madrigal 3500 ROTI-proxy` (±2° quanh trạm, bước 5 phút, `N=862`):**
  - Tổng 4 trạm: **Pearson = `0.037`**, **Spearman = `0.176`** (`HKSL`: `0.348`, `PIMO`: `0.273`, `NTUS`: `0.131`, `CUSV`: `0.016`).
  - Theo giờ (`N=18`): **Pearson = `0.080`**, **Spearman = `-0.015`**.
  - `Madrigal 3500 ROTI-proxy` vs `scnLv2 s4_L1` (`N=14`): **Pearson = `0.147`**, **Spearman = `0.090`**.
- **Kết luận sống còn:** `ROTI-proxy` từ lưới 5 phút `3500` **không đại diện cho ROTI chuẩn 30s** (do bước 5 phút làm trơn mất nhiễu quy mô nhỏ và dính nhiễu nội suy). `3500` chỉ nên dùng cho **bản đồ nền `TEC`** và $\Delta\text{TEC}$; còn **nhãn/đặc trưng `ROTI` phải lấy từ IGS 30s (BKG) kết hợp $S_4$ của `COSMIC-2 scnLv2`**.

## Mốc 5: Tiền tố 3–6h (`OMNI2` + `ESA GIM IONEX`)
- `2024-05-10`: $F_{10.7}=227.9$, $Dst_{\min}=-339\text{ nT}$, $K_{p,\max}=8.7$, $E_{y,\max}=24.39\text{ mV/m}$, EIA chiều (`14–17h LT`) `+21.1 TECU`
- `2024-05-11`: $F_{10.7}=218.0$, $Dst_{\min}=-406\text{ nT}$, $K_{p,\max}=9.0$, $E_{y,\max}=26.19\text{ mV/m}$, EIA chiều `-20.3 TECU`
- `2024-06-04`: $F_{10.7}=198.0$, $Dst_{\min}=-23\text{ nT}$, $K_{p,\max}=2.7$, $E_{y,\max}=1.59\text{ mV/m}$, EIA chiều `+12.4 TECU`

## Mốc 6: Batch đợt 1 Xuân phân 2024 (`2024-03-01` → `2024-04-30`, 58/61 ngày)
- **Dung lượng kho sạch (`data/dataset/`):** `hourly_features.parquet` (**116 KB**) + `scnlv2_points.parquet` (**22.65 MB**). *(3 ngày `03-13`, `03-16`, `03-18` trả 404 trên UCAR `scnLv2_rapid`.)*
- **Benchmark 6 tầm (`1h..6h`) trên `target=regional, S4>=0.50` (`Train=218h (+165)` / `Test=104h (+80)`):**
  - **PR-AUC của `XGBoost` vượt `Persistence` và `Climatology` ở cả 6 tầm:** `1h: 0.936` (vs `0.796`/`0.759`), `2h: 0.898`, `3h: 0.922`, `4h: 0.892`, `5h: 0.892`, `6h: 0.902`.
  - **TSS `XGBoost`:** `1h: 0.488`, `2h: 0.500`, `3h: 0.304`, `4h: 0.325`, `5h: 0.204`, `6h: 0.058` (so với `Climatology = 0.000`).
  - **Brier `XGBoost`:** `0.166 – 0.195` (tốt nhất tại `6h = 0.166`).

## Mốc 7: Mở rộng 3 mùa năm 2024 (Xuân phân + Hạ chí + Thu phân = 174 ngày)
- **Dung lượng kho sạch (`data/dataset/`):** `hourly_features.parquet` (**273 KB**) + `scnlv2_points.parquet` (**43.20 MB**).
- **Benchmark 6 tầm (`1h..6h`) với ngưỡng TSS tối ưu trên Validation:**
  - `target=regional, S4>=0.30` (`Train=572h (+443)` / `Test=256h (+230)`): `XGBoost` đạt PR-AUC **`0.926–0.977`** và Brier **`0.065–0.098`** (vượt cả `Persistence` PR-AUC `0.897–0.905` và `Climatology` `0.904` ở cả 6 tầm). TSS cao nhất ở `1h` (`0.218`) và `2h` (`0.261`).
  - `target=north, S4>=0.30` (`Train=295h (+193)` / `Test=122h (+108)`): `XGBoost` đạt PR-AUC **`0.931`** / TSS **`0.312`** tại `1h` và PR-AUC **`0.900`** / TSS **`0.161`** tại `6h`.
- **Chuyển dịch cơ chế vật lý (Top-10 Feature Importance):**
  - Tại **`lead = 1h` (Nowcast):** `IGS 30s ROTI` dẫn đầu (`HKSL`, `CUSV`, `PIMO`).
  - Tại **`lead = 6h` (Forecast từ chiều):** Nhóm tiền tố vật lý chiếm lĩnh (`cos/sin DOY`, `eia_north`, `eia_south`, `v_sw`, `dst`, `kp`, `f107`, `bz_gsm`).

## Mốc 8: Mô hình Hiệu chuẩn (`XGBoost-Cal` vs `LightGBM-Cal`) trên 174 ngày (`S4 >= 0.50`)
- Chạy [[07-Mo-hinh-va-Hieu-chuan|`src/train_models.py`]] với hiệu chuẩn `sigmoid` trên Validation:
  - `target=vn` (`Train=458 (+275)` / `Test=202 (+143)`): `LightGBM-Cal` đạt PR-AUC **`0.805`** (`1h`), **`0.819`** (`4h`), **`0.797`** (`5h`, TSS `0.174`, Brier `0.188`); `XGBoost-Cal` đạt PR-AUC **`0.797`**, TSS **`0.178`**, Brier **`0.187`** tại `1h` và TSS **`0.180`** tại `3h`.
  - `target=north` (`Train=295 (+193)` / `Test=122 (+108)`): tại `1h`, `XGBoost-Cal` đạt TSS **`0.280`**, Brier **`0.202`**; `LightGBM-Cal` đạt PR-AUC **`0.799`**.
- Đang chạy kéo tiếp 275 ngày (`2024-12`, `2025-03..04`, `2025-09..10`, `2026-03..04`, `2026-09..10`) để hoàn tất Train `2024–2025` & Blind Test `2026`.

## Mốc 9: Bộ dữ liệu Đa năm 2024–2026 (439 ngày) & Kiểm thử mù 2026 (`S4 >= 0.50`)
- **Kho sạch (`data/dataset/`):** **439 ngày** (`2024`: `203`d, `2025`: `115`d → Train/Val; `2026`: `121`d → Blind Test). Dung lượng: `hourly_features.parquet` (**0.58 MB**) + `scnlv2_points.parquet` (**59.25 MB**).
- **Kết quả Kiểm thử mù 2026 (`target=vn`, Train `969h (+607)` / Blind Test `293h (+191)`):**
  - `1h`: `LightGBM-Cal` PR-AUC **`0.784`**, `XGBoost-Cal` PR-AUC **`0.782`**, TSS **`0.224`** (vượt `Climatology` `0.714 / 0.149` và `Persistence` `0.659 / 0.032`).
  - `2h`: `LightGBM-Cal` PR-AUC **`0.753`**, `XGBoost-Cal` TSS **`0.202`** (vượt `Climatology` `0.714 / 0.149`).
  - `3h–6h`: `XGBoost-Cal` PR-AUC **`0.703–0.722`**, Brier **`0.230–0.240`** (tương đương `Climatology` `0.714 / 0.213`).
- **3 miền VN (`S4 >= 0.50`):**
  - `north` (`18–24°N`): `XGBoost-Cal` đạt PR-AUC **`0.763`** (`1h`), **`0.710`** (`6h`).
  - `south` (`6–12°N`): `LightGBM-Cal` đạt PR-AUC **`0.633`** (`1h`), **`0.625`** / TSS **`0.182`** (`3h`).
- **Nhận xét khoa học:** Ở `1–2h`, mạng trạm IGS 30s giúp ML vượt hẳn baseline; ở `3–6h` khi chuyển sang kiểm thử mù 2026 (chu kỳ Mặt Trời suy giảm nhẹ so với 2024–2025 + chỉ còn 3/6 vệ tinh COSMIC-2), `Climatology` theo giờ + $F_{10.7}$ là đối chuẩn rất mạnh, và mô hình đã lưu sẵn trong `models/` để soi trực tiếp trên [[06-Dashboard-va-Van-hanh|Dashboard]].

## Mốc 10: Mô hình lai `Climatology + XGBoost Residual` (`base_margin`) & Bộ hình 300 DPI
- **Cải thiện tầm 3h–6h trên tập kiểm thử mù 2026 (`S4 >= 0.50`):**
  - `target=vn`: `XGB-Resid` đạt PR-AUC **`0.718`** (`3h`), **`0.751`** (`4h`), **`0.726`** (`5h`) — đều vượt `Climatology` (`0.713`) và `XGBoost-Cal` thuần (`0.703–0.722`).
  - `target=central`: `XGB-Resid` đạt PR-AUC **`0.755`** (`2h`), **`0.711`** (`3h`), **`0.691`** (`4h`, TSS `0.156`), **`0.695`** (`5h`, TSS `0.164`), **`0.686`** (`6h`) — vượt `Climatology` (`0.673`) ở cả 5 tầm `2h–6h`.
  - `target=south`: `XGB-Resid` đạt PR-AUC **`0.634`** (`1h`), **`0.626`** (`5h`).
- **Đã xuất bộ 5 hình 300 DPI (`reports/figures/`)** qua `src/export_figures.py`: `fig1_pr_curves_vn.png`, `fig2_calibration_reliability.png`, `fig3_treeshap_1h_vs_6h.png`, `fig4_2026_blind_cases.png`, `fig5_keograms_gannon.png`.
- **Đã đóng gói Reproducibility:** `requirements.txt` + `README.md` chạy lại trọn bộ từ `.parquet`.
