"""Bản đồ 3D WebGL phong cách Zoom Earth (Satellite + City Labels + 3D COSMIC-2) bằng PyDeck."""
import numpy as np
import pandas as pd
import pydeck as pdk

CARTO_DARK_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json"
ESRI_SAT_EXPORT = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
    "?bbox=90,-2,126,30&bboxSR=4326&imageSR=4326&size=1400,1200&format=jpg&f=image"
)

CITIES = [
    {"name": "Ha Noi", "position": [105.85, 21.03, 5000.0]},
    {"name": "Vinh", "position": [105.68, 18.67, 5000.0]},
    {"name": "Da Nang", "position": [108.22, 16.07, 5000.0]},
    {"name": "Nha Trang", "position": [109.19, 12.24, 5000.0]},
    {"name": "TP.HCM", "position": [106.66, 10.76, 5000.0]},
    {"name": "Phu Quoc", "position": [103.96, 10.23, 5000.0]},
    {"name": "Hoang Sa", "position": [112.00, 16.50, 5000.0]},
    {"name": "Truong Sa", "position": [115.83, 10.72, 5000.0]},
]


def _prob_rgba(prob: float, alpha: int = 70) -> list[int]:
    if prob >= 0.65:
        return [220, 38, 38, alpha]
    if prob >= 0.35:
        return [245, 158, 11, alpha]
    return [16, 185, 129, alpha]


def _s4_rgb(s4: float) -> list[int]:
    if not np.isfinite(s4):
        return [148, 163, 184, 160]
    if s4 >= 0.50:
        return [255, 50, 50, 235]
    if s4 >= 0.25:
        return [255, 190, 30, 220]
    return [34, 211, 238, 190]


def build_zoom_earth_deck(
    band_probs: dict[str, float] | None = None,
    c2_df: pd.DataFrame | None = None,
    pitch: float = 35.0,
    bearing: float = 0.0,
    z_scale_m_per_km: float = 450.0,
) -> pdk.Deck:
    """Tạo PyDeck 3D: nền vệ tinh Esri + nhãn thành phố + 3 miền + Dip Equator/EIA + COSMIC-2 3D."""
    layers = [
        pdk.Layer(
            "BitmapLayer",
            id="esri-sat-bg",
            image=ESRI_SAT_EXPORT,
            bounds=[90.0, -2.0, 126.0, 30.0],
            opacity=0.85,
        ),
    ]

    bands = [
        ("north", "Bắc Bộ (18°–24°N)", 18.0, 24.0),
        ("central", "Trung Bộ (12°–18°N)", 12.0, 18.0),
        ("south", "Nam Bộ (6°–12°N)", 6.0, 12.0),
    ]
    poly_rows = []
    for key, label, y0, y1 in bands:
        p = band_probs.get(key, float("nan")) if band_probs else float("nan")
        if np.isfinite(p):
            lvl = "ĐỎ (Rủi ro cao)" if p >= 0.65 else ("VÀNG (Nhiễu vừa)" if p >= 0.35 else "XANH (An toàn)")
            v_str = f"Xác suất EPB: {p*100:.1f}%"
            e_str = f"Cảnh báo: {lvl}"
            fill = _prob_rgba(p, alpha=68)
            line = _prob_rgba(p, alpha=220)
        else:
            v_str = "Khung vùng giám sát Việt Nam"
            e_str = "Theo dõi nhiễu tầng F"
            fill = [34, 197, 94, 20]
            line = [74, 222, 128, 190]

        poly_rows.append({
            "title": label,
            "value_str": v_str,
            "extra_str": e_str,
            "polygon": [[101.0, y0], [116.0, y0], [116.0, y1], [101.0, y1], [101.0, y0]],
            "fill_color": fill,
            "line_color": line,
        })

    layers.append(pdk.Layer(
        "PolygonLayer",
        id="vn-bands",
        data=poly_rows,
        get_polygon="polygon",
        get_fill_color="fill_color",
        get_line_color="line_color",
        line_width_min_pixels=2,
        pickable=True,
    ))

    ref_paths = [
        {
            "title": "Đường Xích đạo từ (~9°N)",
            "value_str": "Nơi khởi phát bong bóng plasma",
            "extra_str": "Chạy qua khu vực Nam Bộ",
            "path": [[95.0, 9.0, 1500], [122.0, 9.0, 1500]],
            "color": [0, 240, 255, 230],
        },
        {
            "title": "Đỉnh Dị thường Điện ly EIA (~20°N)",
            "value_str": "Vùng mật độ điện tử nền cao nhất",
            "extra_str": "Chạy qua khu vực Bắc Bộ",
            "path": [[95.0, 20.0, 1500], [122.0, 20.0, 1500]],
            "color": [232, 121, 249, 210],
        },
    ]
    layers.append(pdk.Layer(
        "PathLayer",
        id="phys-lines",
        data=ref_paths,
        get_path="path",
        get_color="color",
        width_min_pixels=2,
        pickable=True,
    ))

    # Nhãn địa danh Việt Nam rõ nét, chống mất/chồng chữ khi zoom in/out
    layers.append(pdk.Layer(
        "TextLayer",
        id="city-labels",
        data=CITIES,
        get_position="position",
        get_text="name",
        get_size=13,
        size_units="pixels",
        size_min_pixels=11,
        size_max_pixels=16,
        get_color=[255, 255, 255, 255],
        billboard=True,
        parameters={"depthTest": False},
        get_text_anchor="'middle'",
        get_alignment_baseline="'center'",
        pickable=False,
    ))

    if c2_df is not None and not c2_df.empty:
        pts = c2_df[["lon", "lat", "occheight_km", "s4_L1"]].dropna().copy()
        pts["z_m"] = pts["occheight_km"] * z_scale_m_per_km
        pts["color"] = [_s4_rgb(v) for v in pts["s4_L1"]]
        pts["radius"] = 14000 + 28000 * np.clip(pts["s4_L1"], 0, 1.0)
        pts["title"] = "Vệ tinh COSMIC-2 (Tầng F)"
        pts["value_str"] = [f"Chỉ số nhấp nháy S4: {v:.2f}" for v in pts["s4_L1"]]
        pts["extra_str"] = [f"Độ cao đo: {h:.0f} km" for h in pts["occheight_km"]]

        layers.append(pdk.Layer(
            "ScatterplotLayer",
            id="cosmic2-3d",
            data=pts,
            get_position=["lon", "lat", "z_m"],
            get_fill_color="color",
            get_radius="radius",
            radius_min_pixels=3,
            radius_max_pixels=12,
            pickable=True,
        ))

    view = pdk.ViewState(latitude=15.0, longitude=108.5, zoom=4.6, pitch=pitch, bearing=bearing)
    return pdk.Deck(
        layers=layers,
        initial_view_state=view,
        map_style=CARTO_DARK_STYLE,
        tooltip={"html": "<b>{title}</b><br/>{value_str}<br/>{extra_str}"},
    )
