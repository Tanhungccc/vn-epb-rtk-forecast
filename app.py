"""Web App Dự báo Tầng điện ly & Cảnh báo EPB / RTK-GNSS Việt Nam (Streamlit + Plotly + PyDeck).

Chạy:
  .venv/Scripts/streamlit run app.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import fetch_madrigal, live_forecast, map3d, roti, train_models  # noqa: E402

st.set_page_config(page_title="VN EPB & RTK-GNSS Forecast", layout="wide")
st.title("Hệ thống Dự báo Nhiễu Tầng điện ly & Cảnh báo RTK-GNSS / UAV Việt Nam")

FEATURE_LABELS_VI = {
    "utc_hour": "Khung giờ trong ngày",
    "sin_doy": "Chu kỳ mùa trong năm (sin)",
    "cos_doy": "Chu kỳ mùa trong năm (cos)",
    "cos_2doy": "Chu kỳ phân điểm / chí điểm (cos 2·doy)",
    "ey_int_aft": "Tích phân điện trường Ey buổi chiều",
    "dBz_3h": "Biến thiên từ trường Bz trong 3h",
    "dDst_3h": "Biến thiên chỉ số bão Dst trong 3h",
    "dEy_3h": "Biến thiên điện trường Ey trong 3h",
    "dEIA_3h": "Biến thiên chênh lệch TEC EIA trong 3h",
    "ey_x_cos2doy": "Tương tác Ey × Mùa phân điểm",
    "bz_x_cos2doy": "Tương tác Bz × Mùa phân điểm",
    "f107": "Cường độ bức xạ Mặt Trời (F10.7)",
    "bz_gsm": "Từ trường liên hành tinh (Bz)",
    "v_sw": "Tốc độ gió Mặt Trời (v_sw)",
    "ey_mvm": "Điện trường gió Mặt Trời (Ey)",
    "kp": "Chỉ số bão địa từ (Kp)",
    "dst": "Chỉ số bão từ xích đạo (Dst)",
    "eia_north": "Mật độ điện tử đỉnh EIA Bắc Bộ",
    "eia_south": "Mật độ điện tử vùng Nam Bộ",
    "eia_diff": "Chênh lệch TEC Bắc – Nam",
    "eia_ratio": "Tỷ số TEC Bắc / Nam",
    "igs_hksl_roti": "Nhiễu GNSS tại Hồng Kông (HKSL)",
    "igs_cusv_roti": "Nhiễu GNSS tại Bangkok (CUSV)",
    "igs_pimo_roti": "Nhiễu GNSS tại Manila (PIMO)",
    "igs_ntus_roti": "Nhiễu GNSS tại Singapore (NTUS)",
}


def human_feature_name(col: str) -> str:
    if "_t_minus_" in col:
        base, lag = col.split("_t_minus_")
        return f"{FEATURE_LABELS_VI.get(base, base)} ({lag} trước)"
    return FEATURE_LABELS_VI.get(col, col)


@st.cache_data(ttl=900, show_spinner="Đang kéo dữ liệu thời gian thực từ NOAA SWPC & ESA...")
def get_live_bundle():
    return live_forecast.run_live_forecast()


@st.cache_data(show_spinner=True)
def load_historical_dataset():
    feat_p = ROOT / "data" / "dataset" / "hourly_features.parquet"
    c2_p = ROOT / "data" / "dataset" / "scnlv2_points.parquet"
    df_f = pd.read_parquet(feat_p) if feat_p.exists() else pd.DataFrame()
    df_c = pd.read_parquet(c2_p) if c2_p.exists() else pd.DataFrame()
    return df_f, df_c


def rtk_recommendation(prob: float) -> tuple[str, str]:
    if prob >= 0.65:
        return "ĐỎ", "Nguy cơ mất khóa pha / trượt chu kỳ cao sau 19h30. Khuyến cáo tạm dừng đo đạc RTK độ chính xác cao & hạn chế bay UAV."
    if prob >= 0.35:
        return "VÀNG", "Thời gian Fix RTK có thể kéo dài bất thường; hạn chế bay UAV ngoài tầm nhìn (BVLOS)."
    return "XANH", "Hoạt động đo đạc RTK-GNSS và bay UAV bình thường."


with st.expander("ℹ️ Hướng dẫn đọc nhanh cho người dùng (RTK-GNSS / UAV)", expanded=False):
    st.markdown(
        "- **Bong bóng Plasma Xích đạo (EPB):** Là các vùng nhiễu loạn điện ly xuất hiện vào **buổi tối (19:00–24:00 giờ VN)**, có thể làm tín hiệu GPS/RTK bị trễ hoặc mất khóa pha (`S4 >= 0.50`).\n"
        "- **3 mức cảnh báo:** **XANH (< 35%)** = Bình thường | **VÀNG (35%–65%)** = Fix RTK chậm, hạn chế bay UAV xa | **ĐỎ (≥ 65%)** = Nguy cơ mất khóa cao, nên tạm dừng đo đạc độ chính xác cao.\n"
        "- **Cách xoay bản đồ 3D:** Kéo chuột trái để di chuyển; giữ **`Shift` + kéo chuột** (hoặc giữ **chuột phải + kéo**) để xoay/nghiêng góc nhìn 3D; cuộn chuột để phóng to/thu nhỏ."
    )

tab1, tab2, tab3 = st.tabs([
    "1. Dự báo Real-Time Hôm Nay",
    "2. Kiểm thử mù 2026 (COSMIC-2 3D)",
    "3. Phân tích Bão Gannon 2024",
])

# ==================== TAB 1: DỰ BÁO REAL-TIME HÔM NAY ====================
with tab1:
    top_l, top_r = st.columns([4, 1])
    with top_r:
        if st.button("🔄 Cập nhật ngay", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    live = get_live_bundle()
    conf_pct = live.get("data_confidence", 0)
    with top_l:
        st.info(
            f"**{live['mode_label']}** — Giờ hệ thống: **{live['now_vn']}** | "
            f"🎯 **Độ tin cậy dữ liệu: {conf_pct}%**  \n"
            f"📡 **SWPC:** {live['freshness']['swpc']} | "
            f"🌍 **ESA IONEX:** {live['freshness']['ionex']} | "
            f"🛰️ **IGS 30s:** {live['freshness']['igs']}"
        )

    fc = live["forecast_df"]
    if fc.empty:
        st.error("Chưa tìm thấy mô hình trong `models/`. Hãy chạy `python -m src.train_models`.")
    else:
        st.subheader(f"Cảnh báo Vận hành RTK-GNSS & UAV theo 3 Miền (Tin cậy dữ liệu: {conf_pct}%)")
        c_n, c_c, c_s = st.columns(3)
        band_probs = {}
        for col_ui, tgt, label in [
            (c_n, "north", "Bắc Bộ (18°–24°N)"),
            (c_c, "central", "Trung Bộ (12°–18°N)"),
            (c_s, "south", "Nam Bộ (6°–12°N)"),
        ]:
            sub = fc[fc["target"] == tgt]
            p_max = float(sub["prob"].max()) if not sub.empty else 0.0
            peak_lt = sub.loc[sub["prob"].idxmax(), "lt_label"] if not sub.empty else "--"
            band_probs[tgt] = p_max
            lvl, advice = rtk_recommendation(p_max)
            with col_ui:
                msg = f"**{label}** — Mức **{lvl}** (`Cao nhất {p_max*100:.1f}%` lúc `{peak_lt}` | Tin cậy `{conf_pct}%`)  \n{advice}"
                if lvl == "ĐỎ":
                    st.error(msg)
                elif lvl == "VÀNG":
                    st.warning(msg)
                else:
                    st.success(msg)

        dl1, dl2, _ = st.columns([1, 1, 3])
        with dl1:
            st.download_button(
                "⬇️ Tải báo cáo CSV",
                data=fc.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"vn_epb_forecast_{live['now_vn'][:10]}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl2:
            import json as _json
            report_json = {
                "now_vn": live["now_vn"],
                "mode": live["mode_label"],
                "data_confidence_pct": conf_pct,
                "freshness": live["freshness"],
                "swpc": live["swpc"],
                "forecast": fc.to_dict(orient="records"),
            }
            st.download_button(
                "⬇️ Tải báo cáo JSON",
                data=_json.dumps(report_json, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"vn_epb_forecast_{live['now_vn'][:10]}.json",
                mime="application/json",
                use_container_width=True,
            )

        col_map, col_chart = st.columns([1.05, 1.15])
        with col_map:
            st.subheader("Bản đồ Vệ tinh 3D Rủi ro EPB Việt Nam")
            c_p1, c_p2 = st.columns(2)
            pitch_1 = c_p1.slider("Góc nghiêng 3D (°)", 0, 60, 30, key="p1")
            bear_1 = c_p2.slider("Góc xoay Bắc–Nam (°)", -45, 45, 0, key="b1")
            st.pydeck_chart(map3d.build_zoom_earth_deck(band_probs=band_probs, pitch=pitch_1, bearing=bear_1), use_container_width=True, height=430)
            st.caption("Đường xanh cyan: Xích đạo từ (~9°N) | Đường tím: Đỉnh EIA (~20°N). Giữ Shift + kéo để xoay tự do.")

        with col_chart:
            st.subheader("Diễn biến Nguy cơ Nhiễu Toàn Việt Nam (+1h .. +6h)")
            vn_fc = fc[fc["target"] == "vn"].copy()
            vn_fc["pct"] = vn_fc["prob"] * 100.0
            fig_ln = go.Figure()
            fig_ln.add_trace(go.Bar(
                x=vn_fc["lt_label"], y=vn_fc["pct"],
                text=[f"{v:.0f}%" for v in vn_fc["pct"]], textposition="inside",
                textfont=dict(color="white", size=12),
                name="Nguy cơ nhiễu EPB (%)",
                marker_color=["crimson" if p >= 0.65 else ("orange" if p >= 0.35 else "seagreen") for p in vn_fc["prob"]],
            ))
            fig_ln.add_hline(y=65, line_dash="dash", line_color="red")
            fig_ln.add_hline(y=35, line_dash="dot", line_color="orange")
            fig_ln.update_layout(
                yaxis=dict(title="Xác suất xuất hiện EPB (%)", range=[0, 112]),
                xaxis_title="Giờ Việt Nam (LT)",
                height=265, margin=dict(l=10, r=10, t=20, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig_ln, use_container_width=True, config={"displayModeBar": False})
            st.caption("Vạch nét đứt đỏ = 65% (ngưỡng ĐỎ) | Vạch chấm cam = 35% (ngưỡng VÀNG).")

            if live["shap_peak"] is not None and live["shap_peak"][0] is not None:
                sh, lt_p, ld_p, pr_p = live["shap_peak"]
                st.markdown(f"**Yếu tố tác động chính lúc {lt_p:02d}:00 LT (Xác suất {pr_p*100:.1f}%)**")
                top_s = sh.reindex(sh.abs().sort_values(ascending=False).head(5).index)
                fig_s = go.Figure(go.Bar(
                    x=top_s.values,
                    y=[human_feature_name(idx) for idx in top_s.index],
                    orientation="h",
                    marker_color=["crimson" if v > 0 else "royalblue" for v in top_s.values],
                ))
                fig_s.update_layout(
                    xaxis_title="Đỏ: tăng nguy cơ | Xanh: giảm",
                    height=220, margin=dict(l=10, r=20, t=10, b=10),
                )
                st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar": False})

        sw = live["swpc"]
        st.subheader("Chỉ số Thời tiết Không gian Thời gian thực (NOAA SWPC)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Bức xạ Mặt Trời (F10.7)", f"{sw['f107']:.0f} sfu", "Mức nền chu kỳ Mặt Trời")
        m2.metric("Chỉ số Bão từ (Kp)", f"{sw['kp']:.1f}", "≥ 5 là bão từ")
        m3.metric("Tốc độ Gió Mặt Trời", f"{sw['v_sw']:.0f} km/s", "Bình thường ~300–450")
        m4.metric("Từ trường Bz (GSM)", f"{sw['bz_gsm']:.2f} nT", "Âm sâu (< -5) dễ kích EPB")

# ==================== TAB 2: KIỂM THỬ MÙ 2026 ====================
with tab2:
    st.caption("Kiểm thử mù (Prospective Validation): Mô hình chỉ học dữ liệu 2024–2025, chạy dự báo trên năm 2026 và đối chiếu với vệ tinh COSMIC-2 bay qua thực tế.")
    df_f, df_c = load_historical_dataset()
    if df_f.empty:
        st.warning("Chưa có `data/dataset/hourly_features.parquet`.")
    else:
        c_sel1, c_sel2, c_sel3 = st.columns(3)
        years_av = sorted(df_f["year"].unique(), reverse=True)
        sel_y = c_sel1.selectbox("Năm", years_av, key="t2_y")
        sel_d = c_sel2.selectbox("Ngày", sorted(df_f[df_f["year"] == sel_y]["date"].unique(), reverse=True), key="t2_d")
        sel_lead = c_sel3.slider("Tầm dự báo trước (giờ)", 1, 6, 1, key="t2_l")
        sel_tgt = st.radio("Vùng đánh giá", ["vn", "north", "central", "south"], horizontal=True, key="t2_t")

        m_path = ROOT / "models" / f"{sel_tgt}_lead{sel_lead}h.joblib"
        _, s4_col = train_models.COL_MAP[sel_tgt]
        work, _ = train_models.build_lagged(df_f[df_f["date"] == sel_d], sel_lead, s4_col)
        eve = work[(work["utc_hour"] >= 12) & (work["utc_hour"] <= 17)].copy()

        c2_l, c2_r = st.columns([1.15, 1.0])
        with c2_l:
            st.subheader(f"So sánh Dự báo vs Thực đo ({sel_d}, tầm +{sel_lead}h)")
            bundle = joblib.load(m_path)
            probs = bundle["calibrated_model"].predict_proba(eve[bundle["feat_cols"]])[:, 1]
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Bar(
                x=[f"{(h+7)%24:02d}:00 LT" for h in eve["utc_hour"]],
                y=probs * 100.0, name=f"Dự báo ({bundle['model_name'].upper()})",
                marker_color=["crimson" if p >= bundle["threshold"] else "seagreen" for p in probs],
            ))
            has_obs = eve[s4_col].notna()
            if has_obs.any():
                fig_bt.add_trace(go.Scatter(
                    x=[f"{(h+7)%24:02d}:00 LT" for h in eve.loc[has_obs, "utc_hour"]],
                    y=eve.loc[has_obs, s4_col], mode="lines+markers", name="Thực đo S4 max (COSMIC-2)",
                    yaxis="y2", line=dict(color="cyan", width=3),
                ))
            fig_bt.update_layout(
                yaxis=dict(title="Xác suất dự báo (%)", range=[0, 105]),
                yaxis2=dict(title="Chỉ số S4 thực đo (≥0.5 là nhiễu mạnh)", overlaying="y", side="right", range=[0, 1.2]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                height=430, margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_bt, use_container_width=True, config={"displayModeBar": False})

        with c2_r:
            st.subheader(f"Quỹ đạo 3D COSMIC-2 trên nền Vệ tinh ({sel_d})")
            c_p21, c_p22 = st.columns(2)
            pitch_2 = c_p21.slider("Góc nghiêng 3D (°)", 0, 60, 35, key="p2")
            bear_2 = c_p22.slider("Góc xoay Bắc–Nam (°)", -45, 45, 0, key="b2")
            c2_day = df_c[df_c["date"] == sel_d] if not df_c.empty else pd.DataFrame()
            st.pydeck_chart(map3d.build_zoom_earth_deck(c2_df=c2_day, pitch=pitch_2, bearing=bear_2), use_container_width=True, height=390)
            st.caption("Màu điểm vệ tinh: Xanh (S4 < 0.25 êm) | Vàng (0.25–0.50) | Đỏ (≥ 0.50 nhiễu mạnh).")

# ==================== TAB 3: PHÂN TÍCH SỰ KIỆN LỊCH SỬ ====================
with tab3:
    st.subheader("Phân tích Bão Gannon 2024-05-10/11 vs Ngày tĩnh 2024-06-04")
    st.caption("Biểu đồ Keogram giúp quan sát hướng mây plasma trôi từ Tây sang Đông (bên trái) và lan từ Xích đạo từ ~9°N lên Bắc Bộ (bên phải).")
    DAYS_HIST = {
        "2024-05-10 (Bão Gannon G5)": (2024, 5, 10, 131),
        "2024-05-11 (Bão Gannon pha chính)": (2024, 5, 11, 132),
        "2024-06-04 (Ngày tĩnh đối chứng)": (2024, 6, 4, 156),
    }
    sel_h = st.selectbox("Chọn sự kiện", list(DAYS_HIST.keys()), key="t3_d")
    yh, mh, dh, doyh = DAYS_HIST[sel_h]
    p3500 = ROOT / "data" / "raw" / "tec" / f"gps{yh % 100:02d}{mh:02d}{dh:02d}.hdf5"
    if not p3500.exists():
        st.info("Chưa có file Madrigal thô cho sự kiện này.")
    else:
        df_r = roti.compute_roti(roti.compute_rot(fetch_madrigal.load_gridded_tec(p3500)))
        df_r["utc_hour"] = df_r["time"].dt.hour
        sub_h = df_r[(df_r.utc_hour >= 12) & (df_r.utc_hour <= 17) & df_r.gdlat.between(0, 28) & df_r.glon.between(95, 122)]
        k1, k2 = st.columns(2)
        with k1:
            st.subheader("Keogram Kinh độ – Thời gian (Trôi Đông)")
            lon_t = sub_h.groupby(["glon", "time"])["roti"].quantile(0.95).unstack()
            fig_kl = go.Figure(go.Heatmap(
                z=lon_t.values,
                x=[f"{(pd.Timestamp(t).hour+7)%24:02d}:{pd.Timestamp(t).minute:02d}" for t in lon_t.columns],
                y=lon_t.index, colorscale="YlOrRd", zmin=0, zmax=1.5, colorbar=dict(title="ROTI"),
            ))
            fig_kl.update_layout(xaxis_title="Giờ Việt Nam (LT)", yaxis_title="Kinh độ (°E)", height=360, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_kl, use_container_width=True, config={"displayModeBar": False})
        with k2:
            st.subheader("Keogram Vĩ độ – Thời gian (Lan từ Xích đạo từ)")
            lat_t = sub_h.groupby(["gdlat", "time"])["roti"].quantile(0.95).unstack()
            fig_ka = go.Figure(go.Heatmap(
                z=lat_t.values,
                x=[f"{(pd.Timestamp(t).hour+7)%24:02d}:{pd.Timestamp(t).minute:02d}" for t in lat_t.columns],
                y=lat_t.index, colorscale="YlOrRd", zmin=0, zmax=1.5, colorbar=dict(title="ROTI"),
            ))
            fig_ka.add_hline(y=9.0, line_dash="dash", line_color="cyan", annotation_text="Dip Equator (~9°N)")
            fig_ka.update_layout(xaxis_title="Giờ Việt Nam (LT)", yaxis_title="Vĩ độ (°N)", height=360, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_ka, use_container_width=True, config={"displayModeBar": False})
