# Tổng Hợp Nguồn Dữ Liệu Mở Thiên Văn Vô Tuyến & Kỹ Thuật RF (Archival & Live Streams)

> Tài liệu tham chiếu phục vụ nghiên cứu thuật toán **DSP, Xử lý tín hiệu RF, Machine Learning / Deep Learning** cấp độ bài báo khoa học hoặc đồ án chuyên sâu mà **không cần phần cứng ăng-ten vật lý** (Cập nhật hạ tầng & API năm 2026).

---

## Mục Lục
1. [4 Nhóm Kho Dữ Liệu Lưu Trữ Nền Tảng (Archival Data)](#1-4-nhóm-kho-dữ-liệu-lưu-trữ-nền-tảng-archival-data)
2. [7 Nguồn Dữ Liệu Cập Nhật Liên Tục (Real-Time & Daily NRT) Cho Project Thực Nghiệm](#2-7-nguồn-dữ-liệu-cập-nhật-liên-tục-real-time--daily-nrt-cho-project-thực-nghiệm)
3. [3 Tổ Hợp "Ghép Chéo Đa Nguồn" (Cross-Stream Fusion) Dễ Ra Điểm Mới Nhất (2025–2026)](#3-3-tổ-hợp-ghép-chéo-đa-nguồn-cross-stream-fusion-dễ-ra-điểm-mới-nhất-20252026)
4. [Các Lưu Ý Kỹ Thuật & Hạ Tầng Quan Trọng (Cập Nhật 2026)](#4-các-lưu-ý-kỹ-thuật--hạ-tầng-quan-trọng-cập-nhật-2026)
5. [Ma Trận Chọn Nhanh Đề Tài Theo Phần Cứng & Quỹ Thời Gian](#5-ma-trận-chọn-nhanh-đề-tài-theo-phần-cứng--quỹ-thời-gian)
6. [Code Mẫu Python Quick-Start (Zero-Auth — Chạy Thử Ngay)](#6-code-mẫu-python-quick-start-zero-auth--chạy-thử-ngay)

---

## 1. 4 Nhóm Kho Dữ Liệu Lưu Trữ Nền Tảng (Archival Data)

### 1.1. Pulsar, FRB & Dữ liệu Chuỗi Thời gian (Tín hiệu thô & Filterbank)
Nhóm dữ liệu tốt nhất để nghiên cứu thuật toán DSP (lọc nhiễu RFI bằng Spectral Kurtosis, khử tán sắc coherent/incoherent dedispersion, gập pha folding, săn xung đơn/FRB bằng AI).

| Nguồn dữ liệu | Định dạng & Dữ liệu cung cấp | Hướng nghiên cứu thực hiện được ngay |
| :--- | :--- | :--- |
| **CSIRO Data Access Portal (DAP) & CASDA** *(Kính Parkes / Murriyang 64 m & ASKAP)* | Tệp thực `PSRFITS`, filterbank (`.fil`) độ phân giải cao (tự động mở công khai sau thời gian độc quyền 18 tháng) | Tìm kiếm xung đơn / FRB ẩn trong dữ liệu cũ; thử nghiệm thuật toán lọc RFI; nghiên cứu nhấp nháy liên sao (*scintillation*) và cấu trúc vi mô xung pulsar |
| **Breakthrough Listen Open Data Archive** *(GBT 100 m, Parkes, MeerKAT)* | Tệp `.fil`, `.h5` (HDF5) và dữ liệu điện áp thô (*raw baseband* `.raw`) dung lượng rất lớn | Phát triển pipeline khử nhiễu RFI đô thị/vệ tinh bằng Machine Learning; săn tín hiệu nhân tạo (*technosignatures*) hoặc phát hiện FRB băng rộng |
| **CHIME/FRB Public Data** *(400–800 MHz)* | Catalog (Catalog 1 & **Catalog 2 mới 01/2026**) + phổ động (*dynamic spectra / waterfall* HDF5) của hàng nghìn FRB | Phân loại hình thái FRB lặp lại (*repeaters*) vs. không lặp; mô hình hóa tán xạ đa đường (*scattering*) và phân bố DM ngoài Ngân Hà |
| **NANOGrav / IPTA Data Releases** *(Bộ dữ liệu 15 năm)* | Tệp tham số `.par`, thời điểm đến xung `.tim` (TOA), chuỗi dư định thời | Phân tích nhiễu đỏ (*red noise*), biến thiên mật độ electron liên sao ($\delta\text{DM}$ theo thời gian), tái lập phép đo nền sóng hấp dẫn tần số nano-Hz |
| **ATNF Pulsar Catalogue & Jodrell Bank Glitch Catalogue** | Cơ sở dữ liệu tham số ($P, \dot{P}, \text{DM}, S_{1400}$) và lịch sử trượt tần số quay (*glitch* — >737 sự kiện ở >226 pulsar) | Thống kê quần thể pulsar, phân tích quy luật phân bố năng lượng và thời gian chờ (*waiting-time point processes*) giữa các cú glitch của Vela / pulsar trẻ |

---

### 1.2. Khí HI 21 cm, Vạch Phổ & Khảo sát Thiên hà Toàn bầu trời

| Nguồn dữ liệu | Định dạng & Dữ liệu cung cấp | Hướng nghiên cứu thực hiện được ngay |
| :--- | :--- | :--- |
| **HI4PI Survey** *(CDS VizieR / Argelander-Institut)* | Khối dữ liệu 3D FITS $(l, b, v_{\text{LSR}})$ toàn bầu trời ghép từ EBHIS và GASS | Phân tích cấu trúc tay xoắn ốc, độ cong vênh (*warp*) đĩa Ngân Hà, thống kê đám mây vận tốc cao (HVC) và làm chuẩn mô phỏng chùm tia cho kính nhỏ |
| **SALSA Open Dataset** *(Zenodo — Onsala 2.3 m)* | 579 phổ HI chuẩn của kính đơn 2.3 m (kèm hệ thống đăng ký điều khiển kính 2.3 m từ xa miễn phí qua trình duyệt) | Lập đường cong quay Ngân Hà, ước lượng khối lượng vật chất tối, xây bộ dữ liệu đối chuẩn cho phần mềm phân tích phổ |
| **Parkes L-band RRL Survey Archive** *(Jodrell Bank)* | Bản đồ FITS vạch tái hợp vô tuyến $\text{H166}\alpha\text{–H168}\alpha$ và nền liên tục 1.4 GHz dọc mặt phẳng Ngân Hà | Tách thành phần bức xạ nhiệt (*free-free*) và phi nhiệt (*synchrotron*), tính nhiệt độ điện tử $T_e^*$ của các vùng HII |
| **WALLABY (ASKAP) / ALFALFA / THINGS (VLA)** | Khối phổ HI 3D độ phân giải cao của hàng nghìn thiên hà ngoài Ngân Hà | Dựng đường cong quay thiên hà ngoài, kiểm chứng hệ thức Tully–Fisher, nghiên cứu tương tác thủy triều giữa các thiên hà |
| **VLASS / NVSS (VLA), LoTSS (LOFAR), GLEAM (MWA), ALMA Archive** | Ảnh FITS continuum đa tần và dữ liệu giao thoa (*MeasurementSet*) | Lập bản đồ chỉ số phổ ($\alpha$) phân loại tàn dư siêu tân tinh (SNR) / nhân thiên hà hoạt động (AGN); dùng Computer Vision phân loại hình thái tia phản lực vô tuyến (FR-I / FR-II / bent-tail AGN) |

---

### 1.3. Mặt Trời Vô tuyến & Thời tiết Không gian

| Nguồn dữ liệu | Định dạng & Dữ liệu cung cấp | Hướng nghiên cứu thực hiện được ngay |
| :--- | :--- | :--- |
| **Mạng lưới e-CALLISTO** *(>100 trạm phổ kế toàn cầu)* | Phổ động FITS thời gian thực (`.fit.gz`) và kho lưu trữ từ 2002 đến nay ở dải 20–870 MHz | Tự động phát hiện và phân loại vụ nổ Mặt Trời Type II / III / IV bằng Deep Learning; tính vận tốc sóng xung kích của vụ phun trào vành nhật hoa (CME) |
| **Learmonth / RSTN Solar Archive** *(BOM Úc / NOAA)* | Chuỗi thông lượng Mặt Trời nhịp 1 giây tại 8 tần số rời rạc (245–15400 MHz, gồm 1415 MHz) | Phân tích dao động bán chu kỳ (QPP), thống kê phân bố năng lượng bùng phát vô tuyến và đối chiếu tương quan trễ pha với tia X vệ tinh GOES |
| **Nobeyama (NoRH) & Nançay (NRH) Radioheliograph** | Ảnh đĩa Mặt Trời ở 17/34 GHz và băng mét (150–450 MHz) | Kết hợp với ảnh cực tím NASA SDO/AIA để truy vết vùng gia tốc hạt năng lượng cao và cấu trúc từ trường vành nhật hoa |

---

### 1.4. Khí quyển, Tầng Điện ly, Vệ tinh & Mảng Kỹ thuật RF gắn với Việt Nam

* **SatNOGS Network (Libre Space Foundation):** Hàng triệu lượt thu vệ tinh LEO/MEO (ảnh phổ waterfall `.png`, âm thanh giải điều chế `.ogg`, khung dữ liệu giải mã telemetry, và mẫu IQ ở nhiều trạm) $\rightarrow$ Định danh vân tay RF, xác định quỹ đạo từ dịch chuyển Doppler, khảo sát nhiễu phổ vệ tinh.
* **NASA CDDIS (GNSS-TEC / IONEX) & CEDAR Madrigal Database:** Bản đồ tổng mật độ electron tầng điện ly (TEC) theo giờ phủ kín Đông Nam Á $\rightarrow$ Nghiên cứu Dị thường Điện ly Xích đạo (*Equatorial Ionization Anomaly — EIA*) và bong bóng plasma đêm qua Việt Nam.
* **Kho Bóng Thám Không Đại học Wyoming (Upper Air Soundings):** Dữ liệu đo thẳng đứng thực tế hằng ngày (00Z / 12Z) ngay tại trạm **Hà Nội / Nội Bài (`48820`)** và **Tân Sơn Nhất (`48900`)** $\rightarrow$ Tính lượng hơi nước kết tủa (PWV), mô phỏng độ mờ khí quyển ở 22–26 GHz và dự báo suy hao truyền sóng vệ tinh theo khí hậu nhiệt đới gió mùa Việt Nam.
* **Hệ thống thu SDR công cộng (KiwiSDR / WebSDR / OpenWebRX):** Truy cập trực tuyến luồng phổ/âm thanh/IQ thời gian thực từ hàng trăm bộ thu châu Á – thế giới $\rightarrow$ Nghiên cứu truyền sóng tầng điện ly (HF), bão từ gây mất sóng ngắn (*Sudden Ionospheric Disturbance — SID*) và RFI.
* **Quan sát từ xa miễn phí ("tự tay lấy dữ liệu mới"):**
  * **SALSA (Onsala Space Observatory, Thụy Điển):** Đăng ký tài khoản miễn phí điều khiển trực tiếp kính vô tuyến 2.3 m qua trình duyệt để đo vạch HI 21 cm.
  * **Skynet Robotic Telescope Network:** Chương trình giáo dục/nghiên cứu cho phép đặt lịch kính vô tuyến 20 m tại Green Bank.

---

## 2. 7 Nguồn Dữ Liệu Cập Nhật Liên Tục (Real-Time & Daily NRT) Cho Project Thực Nghiệm

Khi làm đề tài thuật toán, dùng nguồn cập nhật liên tục cho phép áp dụng **kiểm định hướng tương lai (prospective out-of-time validation)**: chốt mô hình tại thời điểm $T_0$ và kiểm thử mù trên dữ liệu mới sinh ra ở tuần kế tiếp — loại bỏ triệt để hiện tượng *data leakage*.

### Nguồn 1: e-CALLISTO + NOAA SWPC GOES-XRS (Real-time: 1–15 phút/lần)
* **Nhịp cập nhật:**
  * `e-CALLISTO`: Upload tệp `.fit.gz` mới **mỗi 15 phút** (vào phút `:01, :16, :31, :46` mỗi giờ).
  * `NOAA SWPC JSON`: Cập nhật thông lượng tia X GOES (`0.05–0.4 nm` & `0.1–0.8 nm`) và bùng phát Mặt Trời **mỗi 1 phút**.
* **Dung lượng & Định dạng:** FITS nén (`~0.5–1.5 MB/file 15 phút`) + JSON (`~600 KB – 4 MB`). Không cần API key.
* **Endpoint & Công cụ:**
  * Kho FITS trực tiếp: `http://soleil80.cs.technik.fhnw.ch/solarradio/data/2002-20yy_Callisto/`
  * Tia X thời gian thực: `https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json`
  * Danh sách flare mới nhất: `https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json`
  * Thông lượng vô tuyến Mặt Trời: `https://services.swpc.noaa.gov/json/solar-radio-flux.json`
  * Thư viện Python: `pip install sunpy radiospectra astropy pandas`
* **Project gợi ý:** *Tự động phát hiện vụ nổ Mặt Trời Type II/III và ước lượng vận tốc phóng CME thời gian thực.*
  1. **Input:** Phổ động từ cụm trạm châu Á – Úc (Ấn Độ, Malaysia/Indonesia, Úc — ban ngày trùng múi giờ Việt Nam) + chuỗi tia X GOES.
  2. **DSP:** Khử nền kênh tĩnh (FM/DVB-T/hàng không) bằng *Spectral Kurtosis* hoặc *Sliding-Window Median Subtraction*; cân bằng đáp ứng tần số.
  3. **ML / Vật lý:** Dùng YOLOv8 / U-Net hoặc biến đổi Hough/Radon tách quỹ đạo trôi tần số $df/dt$, áp mô hình mật độ electron vành nhật hoa (Newkirk / Saito) để suy ra vận tốc sóng xung kích CME ($\text{km/s}$).

---

### Nguồn 2: SatNOGS Network & Telemetry DB (Real-time: Từng phút theo lượt vệ tinh bay qua)
* **Nhịp cập nhật:** Liên tục mỗi phút (hàng nghìn lượt thu vệ tinh LEO/MEO mới mỗi ngày từ mạng lưới trạm mặt đất toàn cầu).
* **Dung lượng & Định dạng:** Ảnh thác nước phổ (`.png`), âm thanh băng gốc sau giải điều chế (`.ogg`, ~2–5 MB/lượt), gói tin giải mã hex, và mẫu IQ (`SigMF` / raw tại các trạm cấu hình lưu IQ).
* **Endpoint & Công cụ:**
  * Network REST API: `https://network.satnogs.org/api/observations/`
  * Telemetry & Transmitters DB: `https://db.satnogs.org/api/`
  * TLE cập nhật liên tục: `https://celestrak.org/NORAD/elements/`
  * Thư viện Python: `pip install satnogs-api-client sgp4 skyfield scipy`
* **Project gợi ý:** *Xác định quỹ đạo độc lập từ dịch chuyển Doppler (Doppler Orbit Determination) & Phát hiện vệ tinh cơ động.*
  1. **Input:** File audio `.ogg` / waterfall + tọa độ trạm thu mặt đất từ SatNOGS API + TLE tham chiếu từ CelesTrak.
  2. **DSP:** Ước lượng tần số tức thời $f(t)$ trong điều kiện SNR thấp bằng *PLL số*, *STFT reassignment* hoặc *Viterbi Spectrogram Ridge Tracker* để trích xuất đường cong chữ S Doppler.
  3. **Thuật toán / ML:** Giải bài toán ngược phi tuyến (Levenberg–Marquardt / Extended Kalman Filter với mô hình SGP4) tách riêng độ trôi thạch anh bộ phát (*transmitter thermal drift*) và sai số vị trí dọc quỹ đạo (*along-track error*), phát hiện thời điểm vệ tinh bật động cơ đổi quỹ đạo.

---

### Nguồn 3: CHIME/FRB Real-time GCN Kafka + Catalog 2 (Real-time Stream + Catalog 2026)
* **Nhịp cập nhật:**
  * Luồng cảnh báo phát hiện mới: **Real-time** qua NASA GCN Kafka topic `gcn.notices.chime.frb`.
  * Cơ sở dữ liệu chuẩn hóa: **Catalog 2** (vừa phát hành **14/01/2026**, DOI: `10.11570/25.0066`) gồm **5,045 sự kiện FRB** và **83 nguồn lặp lại**, kèm REST API JSON mở không cần đăng ký.
* **Dung lượng & Định dạng:** JSON qua REST API / Kafka + tệp phổ động hiệu chuẩn HDF5 (`.h5`, toàn bộ kho nhỏ gọn `< 2 GB` cho dữ liệu rút gọn hoặc tải từng burst trên CANFAR).
* **Endpoint & Công cụ:**
  * REST API (không cần key): `https://chime-frb.ca/api/v1/catalogs/catalog2/bursts`
  * Tài liệu API: `https://chime-frb.ca/tutorials`
  * Stream thời gian thực: `pip install gcn-kafka` (đăng ký `client_id` / `client_secret` miễn phí tự động tại `https://gcn.nasa.gov`).
* **Project gợi ý:** *Phân loại sớm nguồn FRB lặp lại (Repeaters) từ một lần bùng phát duy nhất bằng Hình thái Thời gian – Tần số.*
  1. **Input:** Phổ động 2D (Waterfall) và chuỗi thời gian khử tán sắc của 5,045 bursts trong Catalog 2 + các sự kiện mới từ Kafka.
  2. **DSP:** Khử tán sắc tinh chỉnh tối ưu hóa cấu trúc vi xung (*structure-maximizing DM*), tính hàm tự tương quan 2D (*2D Autocorrelation Function*) đo tốc độ trượt tần số con (*sad-trombone sub-burst drift*) và đuôi tán xạ $\tau_{\text{sc}}$.
  3. **ML:** Vì nguồn "chưa thấy lặp" có thể thực chất là nguồn lặp chưa quan sát đủ lâu, áp dụng **Positive-Unlabeled (PU) Learning** hoặc *Contrastive Self-Supervised Learning* để xếp hạng ứng viên nguồn lặp tiềm năng.

---

### Nguồn 4: WSPR.live Global HF Propagation Stream (Real-time: Mỗi 2 phút)
* **Nhịp cập nhật:** **Mỗi 2 phút** (trễ chỉ vài phút so với thực tế), tổng cộng hơn **4 tỷ bản ghi** truyền sóng HF công suất cực thấp (2–30 MHz) toàn cầu.
* **Dung lượng & Định dạng:** Cơ sở dữ liệu ClickHouse cho phép gửi **câu lệnh SQL trực tiếp qua HTTP GET** trả về JSON/CSV/Parquet, không cần tài khoản, tải cực nhẹ (chỉ lấy đúng vùng/khung giờ cần nghiên cứu).
* **Endpoint & Công cụ:**
  * ClickHouse HTTP SQL: `https://db1.wspr.live/?query=SELECT ... FORMAT JSON`
  * Giao diện tài liệu: `https://wspr.live/`
* **Project gợi ý:** *Giám sát Dị thường Hấp thụ Tầng D (SID / Radio Blackout) & Dự báo kênh truyền sóng ngắn khu vực Đông Nam Á.*
  1. **Input:** Truy vấn các đường truyền có điểm phản xạ tầng điện ly cắt qua Đông Nam Á / Biển Đông (lọc lưới Maidenhead `OK`, `OL`, `OJ` quanh Việt Nam và lân cận) ở các băng 7, 10, 14, 21, 28 MHz.
  2. **DSP / Đặc trưng:** Trích xuất biến thiên `SNR` và `Drift` (độ dịch tần Doppler Hz/phút do chuyển động thẳng đứng của lớp phản xạ F2/E).
  3. **ML:** Xây dựng *Spatio-Temporal Graph Neural Network (ST-GNN)* coi mỗi trạm thu/phát là một nút đồ thị để phát hiện tức thời hiện tượng mất sóng HF do bùng phát tia X Mặt Trời và dự báo tần số khả dụng tối đa (MUF).

---

### Nguồn 5: COSMIC-2 GNSS Radio Occultation + Bóng Thám Không Việt Nam (Hàng ngày — 02:00 UTC)
* **Nhịp cập nhật:**
  * **COSMIC-2 (FORMOSAT-7) Near-Real-Time:** Công bố **hằng ngày trước 02:00 UTC** cho toàn bộ ngày hôm trước. Sáu vệ tinh COSMIC-2 có góc nghiêng quỹ đạo $24^\circ$ nên **quét cực kỳ dày đặc qua dải vĩ độ Việt Nam ($8^\circ\text{N}–24^\circ\text{N}$)**.
  * **Wyoming Upper Air Soundings:** Cập nhật **2 lần/ngày (00Z & 12Z)** tại trạm **Hà Nội (`48820`)** và **Tân Sơn Nhất (`48900`)**.
* **Dung lượng & Định dạng:** Tệp Tar/NetCDF4 (`podTc2` cho TEC & chỉ số nhấp nháy $S_4$, `ionPhs`, `wetPf2` cho nhiệt–ẩm khí quyển); khoảng 50–300 MB/tháng sau khi lọc tọa độ Việt Nam.
* **Endpoint & Công cụ:**
  * Kho mở UCAR NRT: `https://data.cosmic.ucar.edu/gnss-ro/cosmic2/nrt/level2/`
  * Mirror NOAA/UMD: `https://gpsmet.umd.edu/star_gnssro/data/cosmic2/nrt/`
  * Thư viện Python: `pip install xarray netCDF4 siphon metpy`
* **Project gợi ý:** *Dự báo Nhấp nháy Vệ tinh GPS ($S_4$ Scintillation) sau hoàng hôn & Mô hình hóa suy hao truyền sóng K/Ka-band (22–30 GHz) tại Việt Nam.*
  1. **Input:** Các mặt cắt thẳng đứng mật độ electron $N_e(h)$, chỉ số $S_4$ và độ ẩm `wetPf2` trong khung $8^\circ\text{N}–24^\circ\text{N}, 102^\circ\text{E}–112^\circ\text{E}$ + dữ liệu bóng thám không 48820 & 48900.
  2. **DSP / Vật lý khí quyển:** Tính lượng hơi nước kết tủa (PWV) và tích phân suy hao hấp thụ khí quyển theo chuẩn ITU-R P.676 tại vạch hơi nước 22.235 GHz (phục vụ thông tin vệ tinh băng rộng / thiên văn vô tuyến tần số cao tại VN).
  3. **ML:** Huấn luyện XGBoost / Temporal Fusion Transformer dự báo xác suất hình thành **Bong bóng Plasma Xích đạo (Equatorial Plasma Bubbles)** khung giờ 19h–24h tối tại Việt Nam (thủ phạm chính gây mất khóa pha trạm đo đạc RTK-GNSS).

---

### Nguồn 6: CSIRO CASDA — ASKAP VAST + Parkes Rolling Archive (Mở khóa hàng ngày)
* **Nhịp cập nhật:**
  * Chính sách embargo 18 tháng của CSIRO hoạt động theo cơ chế cuốn chiếu (*rolling release*): **mỗi ngày đều có các tệp `PSRFITS` / `.fil` mới của kính Parkes 64 m được tự động mở khóa công khai**.
  * Ngoài ra, khảo sát **ASKAP VAST Extragalactic Survey DR1** (vừa công bố **02/2026**, arXiv:2602.22739) mở toàn bộ **6.4 triệu phép đo đường cong ánh sáng của 0.5 triệu nguồn vô tuyến** ở 888 MHz.
* **Dung lượng & Định dạng:** Bảng biến thiên Parquet/VOTable (~1 GB) hoặc tệp thô `PSRFITS` / `.fil` (10–100+ GB).
* **Endpoint & Công cụ:**
  * TAP Server: `https://casda.csiro.au/casda_vo_tools/tap`
  * Thư viện Python: `pip install astroquery vast-pipeline your` (`from astroquery.casda import Casda`)
* **Project gợi ý:**
  * *Hướng nhẹ máy:* Khai phá kho đường cong ánh sáng VAST DR1 bằng thuật toán phát hiện bất thường (*Isolation Forest / Variational Autoencoder*) kết hợp đối chiếu phân cực tròn để tìm **sao lùn đỏ bùng phát vô tuyến (radio stars)** hoặc **magnetar** chưa được định danh.
  * *Hướng nặng DSP:* Tự động tải các tệp `PSRFITS` băng L (0.7–4 GHz từ đầu thu UWL Parkes) vừa mở khóa để đối chuẩn thuật toán lọc nhiễu vệ tinh định vị (GPS/Galileo/BeiDou) bằng phân rã trị riêng không gian–tần số mà không làm méo xung pulsar/FRB.

---

### Nguồn 7: CEDAR Madrigal GNSS-TEC & INTERMAGNET Việt Nam (Hàng ngày NRT)
* **Nhịp cập nhật:** Bản đồ TEC toàn cầu từ mạng lưới hàng nghìn trạm GPS mặt đất cập nhật hằng ngày (độ trễ NRT 1–3 ngày); số liệu biến thiên từ trường Trái Đất tại **Từ đài Phú Thụy — Hà Nội (`PHU`)** và **Đà Lạt (`DLT`)** thuộc mạng INTERMAGNET cập nhật gần thời gian thực.
* **Dung lượng & Định dạng:** HDF5 / NetCDF4 / ASCII lọc trực tiếp trên server trước khi tải (~200 MB – 2 GB/năm cho khung Đông Nam Á).
* **Endpoint & Công cụ:**
  * Madrigal Server: `http://cedar.openmadrigal.org`
  * Thư viện Python: `pip install madrigalWeb h5py`
* **Project gợi ý:** *Dự báo chuỗi ảnh Không gian – Thời gian (ConvLSTM / Swin-Unet) cho đỉnh Dị thường Điện ly Xích đạo (EIA) trên lãnh thổ Việt Nam.*
  1. **Input:** Chuỗi bản đồ lưới TEC vùng $0^\circ–30^\circ\text{N}, 95^\circ–120^\circ\text{E}$ + thành phần từ trường ngang $H$ tại trạm `PHU` / `DLT` + chỉ số gió Mặt Trời $B_z$, $K_p$, $\text{Dst}$ từ NOAA SWPC.
  2. **DSP:** Nội suy không gian có trọng số vật lý và phân rã đa thang thời gian (*CEEMDAN / Wavelet*) tách chu kỳ ngày đêm khỏi nhiễu động bão từ.
  3. **ML:** Dự báo trước 1–6 giờ bản đồ TEC khu vực Việt Nam trong các trận bão địa từ lớn của chu kỳ Mặt Trời 25, đánh giá cải thiện RMSE so với mô hình quốc tế IRI-2020.

---

## 3. 3 Tổ Hợp "Ghép Chéo Đa Nguồn" (Cross-Stream Fusion) Dễ Ra Điểm Mới Nhất (2025–2026)

1. **Tổ hợp Thời tiết Không gian Xích đạo tại Việt Nam:**
   `NOAA GOES X-ray (1 phút)` + `e-CALLISTO châu Á/Úc (15 phút)` $\longrightarrow$ `WSPR.live HF Đông Nam Á (2 phút)` + `COSMIC-2 qua Việt Nam`.
   * *Giá trị khoa học:* Định lượng trực tiếp độ trễ và biên độ suy hao truyền sóng vĩ độ thấp tại Đông Nam Á ngay khi xảy ra bùng phát Mặt Trời lớp M/X trong đỉnh Solar Cycle 25.

2. **Tổ hợp Cảnh báo & Phân loại Nhanh Nguồn Chuyển tiếp Vật lý Thiên văn:**
   `NASA GCN Kafka (gcn.notices.chime.frb)` $\longrightarrow$ `CHIME/FRB Catalog 2 API (5,045 bursts)`.
   * *Giá trị khoa học:* Xây dựng hệ thống chấm điểm tự động thời gian thực ngay khi có thông báo FRB mới để ưu tiên hóa việc quan sát theo dõi (*follow-up*) cho các kính thiên văn khác.

3. **Tổ hợp Kỹ thuật RF & Giám sát Không gian Độc lập (SSA — Space Situational Awareness):**
   `SatNOGS Live Audio/Waterfall` $\longrightarrow$ `CelesTrak TLE`.
   * *Giá trị khoa học:* Thuần kỹ thuật xử lý tín hiệu số & ước lượng trạng thái (PLL + Kalman/SGP4), phù hợp hoàn hảo cho định hướng Viễn thông / Radar / Kỹ thuật Hàng không – Vũ trụ.

---

## 4. Các Lưu Ý Kỹ Thuật & Hạ Tầng Quan Trọng (Cập Nhật 2026)

* **Ngừng giao thức VOEvent XML truyền thống (06/04/2026):** Mạng GCN Classic VOEvent Transport Protocol đã chính thức nghỉ hưu từ ngày 06/04/2026. Mọi code mẫu cũ trên GitHub dùng `comet` hoặc `voevent-parse` qua socket cổ điển cần thay bằng **`gcn-kafka`** (đọc thông điệp JSON trên Kafka của NASA GCN).
* **Tận dụng đỉnh Chu kỳ Mặt Trời 25 (Solar Cycle 25 Peak, 2024–2026):** Tần suất xuất hiện các vụ nổ vô tuyến Mặt Trời Type II/III/IV, bão địa từ mạnh và bong bóng plasma xích đạo hiện đang ở mức cao nhất trong hơn một thập kỷ, giúp thu thập được hàng trăm sự kiện dương tính (*positive events*) chỉ trong vài tháng dữ liệu thay vì phải gom nhiều năm như thời kỳ Mặt Trời tĩnh.
* **Tham số địa lý chuẩn khi lọc dữ liệu khu vực Việt Nam:**
  * Khung tọa độ đất liền & vùng biển Việt Nam: Vĩ độ **`6°N – 24°N`**, Kinh độ **`101°E – 116°E`**.
  * Ô lưới vô tuyến nghiệp dư (Maidenhead Grid Squares cho WSPR): **`OK`** (Bắc Bộ, Trung Bộ, Biển Đông), **`OL`** (Biên giới phía Bắc), **`OJ`** (Nam Bộ & vùng biển phía Nam).
  * Trạm khí tượng cao không (WMO Station ID): **`48820`** (Hà Nội / Nội Bài), **`48900`** (TP.HCM / Tân Sơn Nhất), **`48855`** (Đà Nẵng).
  * Trạm địa từ quốc tế (INTERMAGNET IAGA Code): **`PHU`** (Phú Thụy, Gia Lâm, Hà Nội), **`DLT`** (Đà Lạt).

---

## 5. Ma Trận Chọn Nhanh Đề Tài Theo Phần Cứng & Quỹ Thời Gian

| Điều kiện thực tế & Mục tiêu cá nhân | Nguồn khuyên dùng số 1 | Dung lượng đĩa cần | Thư viện cốt lõi |
| :--- | :--- | :--- | :--- |
| **Laptop cá nhân / Colab Free**, cần ra kết quả chạy thử trong 2–3 ngày đầu | **e-CALLISTO + NOAA GOES** | `1 – 5 GB` | `sunpy`, `radiospectra`, `ultralytics` / `pytorch` |
| **Định hướng Kỹ thuật RF, SDR, Viễn thông vệ tinh, DSP cứng** | **SatNOGS + CelesTrak** | `2 – 10 GB` | `scipy.signal`, `sgp4`, `satnogs-api-client` |
| **Định hướng Vật lý Thiên văn hiện đại (FRB / Astro-ML)** | **CHIME Catalog 2 + GCN Kafka** | `< 3 GB` | `h5py`, `requests`, `gcn-kafka`, `scikit-learn` |
| **Bài toán thực tiễn gắn trực tiếp với Việt Nam (GNSS, HF, 5G/Vệ tinh)** | **COSMIC-2 + WSPR.live + Madrigal** | `1 – 5 GB` | `xarray`, `netCDF4`, `madrigalWeb`, `pytorch` |
| **Máy trạm ổ cứng lớn ($\ge 100\text{ GB}$), muốn xử lý tín hiệu thô kính lớn** | **CSIRO CASDA (Parkes PSRFITS)** | `100 GB+` | `astroquery`, `your`, `presto` / `sigpyproc` |

---

## 6. Code Mẫu Python Quick-Start (Zero-Auth — Chạy Thử Ngay)

### 6.1. Kéo thông lượng tia X Mặt Trời thời gian thực (NOAA GOES — cập nhật mỗi 1 phút)
```python
import pandas as pd

url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json"
df = pd.read_json(url)
xrs = df.pivot(index="time_tag", columns="energy", values="observed_flux")
xrs.columns = ["xrsa_0.05_0.4nm", "xrsb_0.1_0.8nm"]
print("5 mẫu đo tia X Mặt Trời mới nhất (cách đây vài phút):")
print(xrs.tail())
```

### 6.2. Truy vấn danh mục CHIME/FRB Catalog 2 (vừa công bố 01/2026 — không cần API key)
```python
import requests

resp = requests.get(
    "https://chime-frb.ca/api/v1/catalogs/catalog2/bursts",
    params={"per_page": 5, "view": "repeat"},
    timeout=20,
)
resp.raise_for_status()
data = resp.json()
print("Ví dụ các sự kiện từ nguồn FRB lặp lại trong Catalog 2:")
for row in data.get("data", data if isinstance(data, list) else [])[:5]:
    print(row)
```

### 6.3. Truy vấn SQL trực tiếp các đường truyền sóng HF quanh Việt Nam 1 giờ qua (`wspr.live`)
```python
import urllib.parse
import requests
import pandas as pd

sql = """
SELECT time, band, tx_sign, tx_loc, rx_sign, rx_loc, distance, snr, drift
FROM wspr.rx
WHERE time > now() - INTERVAL 1 HOUR
  AND (substring(tx_loc, 1, 2) IN ('OK', 'OJ', 'OL')
       OR substring(rx_loc, 1, 2) IN ('OK', 'OJ', 'OL'))
ORDER BY time DESC
LIMIT 20
FORMAT JSONCompact
"""
url = "https://db1.wspr.live/?query=" + urllib.parse.quote(sql)
res = requests.get(url, timeout=20).json()
cols = [m["name"] for m in res["meta"]]
df_wspr = pd.DataFrame(res["data"], columns=cols)
print(df_wspr.head(10))
```

### 6.4. Tìm và tải phổ động Mặt Trời e-CALLISTO mới nhất bằng `sunpy` + `radiospectra`
```python
# pip install sunpy radiospectra
from sunpy.net import Fido, attrs as a
import radiospectra.net  # đăng ký client eCALLISTO vào Fido

query = Fido.search(
    a.Time("2026-04-01 02:00", "2026-04-01 03:00"),
    a.Instrument("eCALLISTO"),
)
print(query)
# Tải file đầu tiên về máy:
# files = Fido.fetch(query[0, 0])
```

### 6.5. Lấy danh sách các lượt thu vệ tinh vừa hoàn tất trên SatNOGS Network
```python
import requests

obs = requests.get(
    "https://network.satnogs.org/api/observations/",
    params={"status": "good", "format": "json"},
    timeout=20,
).json()

for item in obs[:5]:
    print(
        f"ID: {item['id']} | NORAD: {item['norad_cat_id']} | "
        f"Waterfall: {item.get('waterfall')} | Audio: {item.get('payload') or item.get('archive_url')}"
    )
```
