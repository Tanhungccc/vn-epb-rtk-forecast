"""Cấu hình chung cho pipeline TEC / ROTI / EPB Việt Nam."""

# --- Khung tọa độ nghiên cứu: lãnh thổ & biển Việt Nam ---
LAT_MIN, LAT_MAX = 6.0, 24.0
LON_MIN, LON_MAX = 101.0, 116.0

# --- Ngưỡng ROTI (proxy từ bản đồ TEC 5 phút): TECU/phút ---
ROTI_THRESHOLD = 0.5

# --- Cửa sổ ROTI: số map 5 phút liên tiếp để tính std(ROT) ---
ROTI_WINDOW_MAPS = 5  # = 20 phút

# --- Madrigal global TEC (World-wide GNSS Receiver Network) ---
MADRIGAL_BASE = "https://cedar.openmadrigal.org/ftp/fullname/Anonymous/email/anonymous@example.com/affiliation/Independent"
MADRIGAL_KINST = 8000   # World-wide GNSS Receiver Network
MADRIGAL_KINDAT = 3500  # TEC binned 1deg x 1deg x 5 min

# --- COSMIC-2 NRT ---
COSMIC2_LEVEL1B = "https://data.cosmic.ucar.edu/gnss-ro/cosmic2/nrt/level1b"

# --- NOAA SWPC (zero-auth JSON) ---
SWPC_KP_1M = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
SWPC_MAG_1M = "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"  # realtime solar wind mag, 1-min, ~1.7 MB

# --- Bão từ mục tiêu (bão Gannon G5, 10-11/05/2024) ---
TARGET_STORM_DAYS = [
    (2024, 5, 10),
    (2024, 5, 11),
]

# --- Tuần tĩnh để đối chứng (quiet reference) ---
QUIET_REF_DAYS = [
    (2024, 6, 4),
]
