"""Dashboard Dự báo & Giám sát Tầng điện ly / EPB Việt Nam (Streamlit + Plotly).

Hỗ trợ 2 chế độ:
  1. Dự báo Mô hình thật (+1h..+6h) & Kiểm thử mù (2024–2026):
     Đọc data/dataset/hourly_features.parquet, scnlv2_points.parquet và models/*.joblib;
     so sánh xác suất dự báo đã hiệu chuẩn với lát cắt 3D COSMIC-2 thật và TreeSHAP.
  2. Case Study chi tiết (Bão Gannon 2024-05-10/11 vs Ngày tĩnh 2024-06-04):
     Hiển thị đầy đủ lưới Madrigal 3500 + IGS 30s + COSMIC-2 3D + Keograms.

Chạy:
  .venv/Scripts/streamlit run src/app_demo.py
"""
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import fetch_madrigal, precursors, roti, scnlv2, train_models  # noqa: E402

st.set_page_config(page_title="VN Ionosphere & EPB Forecast", layout="wide")
st.title("Dự báo & Giám sát Tầng điện ly / EPB Việt Nam (0°–28°N, 95°–122°E)")

mode = st.sidebar.radio(
    "Chế độ hiển thị",
    ["1. Dự báo Mô hình thật (+1h..+6h) & Kiểm thử mù", "2. Case Study chi tiết (Gannon 2024-05 vs Tĩnh 06-04)"],
)


@st.cache_data(show_spinner=True)
def load_dataset():
    feat_p = ROOT / "data" / "dataset" / "hourly_features.parquet"
    c2_p = ROOT / "data" / "dataset" / "scnlv2_points.parquet"
    df_f = pd.read_parquet(feat_p) if feat_p.exists() else pd.DataFrame()
    df_c = pd.read_parquet(c2_p) if c2_p.exists() else pd.DataFrame()
    return df_f, df_c


def _load_bundle(target: str, lead: int):
    p = ROOT / "models" / f"{target}_lead{lead}h.joblib"
    return joblib.load(p) if p.exists() else None


def _shap_contrib(bundle: dict, X_row: pd.DataFrame) -> pd.Series | None:
    base = bundle.get("base_model")
    cols = bundle.get("feat_cols", [])
    if base is None:
        return None
    if hasattr(base, "get_booster"):
        dm = xgb.DMatrix(X_row[cols])
        vals = base.get_booster().predict(dm, pred_contribs=True)[0][:-1]
        return pd.Series(vals, index=cols)
    if hasattr(base, "predict"):
        vals = base.predict(X_row[cols], pred_contrib=True)[0][:-1]
        return pd.Series(vals, index=cols)
    return None


if mode.startswith("1."):
    df_f, df_c = load_dataset()
    if df_f.empty:
        st.warning("Chưa có `data/dataset/hourly_features.parquet`. Hãy chạy `batch_ingest.py` trước.")
        st.stop()

    years_avail = sorted(df_f["year"].unique(), reverse=True)
    sel_year = st.sidebar.selectbox("Chọn năm", years_avail)
    dates_yr = sorted(df_f[df_f["year"] == sel_year]["date"].unique(), reverse=True)
    sel_date = st.sidebar.selectbox("Chọn ngày (Ưu tiên 2026 kiểm thử mù)", dates_yr)
    lead = st.sidebar.slider("Tầm dự báo (lead_hours)", 1, 6, 3)
    target = st.sidebar.selectbox("Vùng mục tiêu", ["vn", "north", "central", "south"])
    show_sporadic = st.sidebar.checkbox("3D: Hiện cả vùng thấp < 200 km (nếu có)", value=False)

    bundle = _load_bundle(target, lead)
    _, s4_col = train_models.COL_MAP[target]
    work, feat_cols = train_models.build_lagged(df_f[df_f["date"] == sel_date], lead, s4_col)
    eve = work[(work["utc_hour"] >= 12) & (work["utc_hour"] <= 17)].copy()

    col_left, col_right = st.columns([1.2, 1.0])
    with col_left:
        st.subheader(f"Dự báo Xác suất EPB (+{lead}h) vs Thực đo COSMIC-2 ({sel_date}, target={target})")
        if bundle is None:
            st.info(f"Chưa có mô hình `models/{target}_lead{lead}h.joblib`. Hãy chạy `python -m src.train_models`.")
        else:
            probs = bundle["calibrated_model"].predict_proba(eve[bundle["feat_cols"]])[:, 1]
            eve["pred_prob"] = probs
            th = bundle["threshold"]
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Bar(
                x=[f"{h:02d} UTC ({(h+7)%24:02d}h VN)" for h in eve["utc_hour"]],
                y=eve["pred_prob"], name=f"Xác suất EPB ({bundle['model_name'].upper()}-Cal)",
                marker_color=["crimson" if p >= th else ("orange" if p >= 0.25 else "seagreen") for p in probs],
            ))
            fig_fc.add_hline(y=th, line_dash="dash", line_color="red", annotation_text=f"Ngưỡng TSS ({th:.2f})")
            # Chồng S4 thực đo nếu có COSMIC-2 qua vùng
            has_obs = eve[s4_col].notna()
            if has_obs.any():
                fig_fc.add_trace(go.Scatter(
                    x=[f"{h:02d} UTC ({(h+7)%24:02d}h VN)" for h in eve.loc[has_obs, "utc_hour"]],
                    y=eve.loc[has_obs, s4_col], mode="lines+markers", name="Thực đo S4 max (scnLv2)",
                    yaxis="y2", line=dict(color="cyan", width=3),
                ))
            fig_fc.update_layout(
                yaxis=dict(title="Xác suất EPB", range=[0, 1.05]),
                yaxis2=dict(title="S4 thực đo", overlaying="y", side="right", range=[0, 1.2]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                height=380, margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            # Giải thích TreeSHAP cho giờ có xác suất cao nhất
            peak_idx = eve["pred_prob"].idxmax()
            contrib = _shap_contrib(bundle, eve.loc[[peak_idx]])
            if contrib is not None:
                top_c = contrib.reindex(contrib.abs().sort_values(ascending=False).head(8).index)
                fig_sh = go.Figure(go.Bar(
                    x=top_c.values, y=top_c.index, orientation="h",
                    marker_color=["crimson" if v > 0 else "royalblue" for v in top_c.values],
                ))
                fig_sh.update_layout(
                    title=f"Đóng góp đặc trưng (TreeSHAP) tại {int(eve.loc[peak_idx, 'utc_hour']):02d} UTC",
                    xaxis_title="Đóng góp log-odds (+ tăng nguy cơ EPB)",
                    height=300, margin=dict(l=10, r=10, t=35, b=10),
                )
                st.plotly_chart(fig_sh, use_container_width=True)

    with col_right:
        st.subheader(f"Quỹ đạo 3D COSMIC-2 thật ({sel_date}, 12–17 UTC)")
        c2_day = df_c[df_c["date"] == sel_date] if not df_c.empty else pd.DataFrame()
        fig3d = go.Figure()
        fig3d.add_trace(go.Scatter3d(
            x=[101, 116, 116, 101, 101], y=[6, 6, 24, 24, 6], z=[0, 0, 0, 0, 0],
            mode="lines", name="Khung VN (z=0)", line=dict(color="green", width=4),
        ))
        fig3d.add_trace(go.Scatter3d(
            x=[95, 122], y=[9, 9], z=[0, 0], mode="lines", name="Dip Equator (z=0)",
            line=dict(color="cyan", width=3, dash="dash"),
        ))
        if not c2_day.empty:
            fig3d.add_trace(go.Scatter3d(
                x=c2_day["lon"], y=c2_day["lat"], z=c2_day["occheight_km"], mode="markers",
                name="Tầng F (≥200 km)",
                marker=dict(size=3 + 8 * c2_day["s4_L1"].fillna(0), color=c2_day["s4_L1"], colorscale="YlOrRd", cmin=0, cmax=0.8),
            ))
        else:
            st.info("Ngày này không có lát cắt COSMIC-2 đi qua khung 0°–28°N, 95°–122°E trong 12–17 UTC.")
        fig3d.update_layout(
            scene=dict(
                xaxis_title="Lon (°E)", yaxis_title="Lat (°N)", zaxis_title="Độ cao (km)",
                zaxis=dict(range=[0, 560]),
                aspectmode="manual", aspectratio=dict(x=1.2, y=1.2, z=0.8),
                camera=dict(eye=dict(x=1.55, y=-1.65, z=0.95)),
            ),
            height=480, margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig3d, use_container_width=True)

        st.subheader("Tiền tố OMNI2 & ESA IONEX trong ngày")
        st.dataframe(eve[["utc_hour", "f107", "bz_gsm", "v_sw", "ey_mvm", "kp", "dst", "eia_diff"]], use_container_width=True)

else:
    DAYS = {
        "2024-05-10 (Bão Gannon G5)": (2024, 5, 10, 131),
        "2024-05-11 (Bão Gannon pha chính/phục hồi)": (2024, 5, 11, 132),
        "2024-06-04 (Ngày tĩnh đối chứng)": (2024, 6, 4, 156),
    }

    @st.cache_data(show_spinner=True)
    def load_day_bundle(year: int, month: int, day: int, doy: int):
        p3500 = ROOT / "data" / "raw" / "tec" / f"gps{year % 100:02d}{month:02d}{day:02d}.hdf5"
        df_tec = fetch_madrigal.load_gridded_tec(p3500)
        df_roti = roti.compute_roti(roti.compute_rot(df_tec))
        df_roti["utc_hour"] = df_roti["time"].dt.hour
        df_roti = df_roti[
            (df_roti.utc_hour >= 12) & (df_roti.utc_hour <= 17) &
            df_roti.gdlat.between(0, 28) & df_roti.glon.between(95, 122)
        ].copy()

        tar_c2 = ROOT / "data" / "raw" / "cosmic2" / f"scnLv2_rapid_{year}_{doy:03d}.tar.gz"
        df_c2, _ = scnlv2.parse_scnlv2_tar(
            tar_c2, lat_min=0.0, lat_max=28.0, lon_min=95.0, lon_max=122.0, min_occheight_km=60.0
        )
        if not df_c2.empty:
            df_c2 = df_c2[(df_c2.utc_hour >= 12) & (df_c2.utc_hour <= 17)].copy()
            df_c2["time_5min"] = df_c2["time"].dt.floor("5min")

        igs_pq = ROOT / "data" / "out" / "igs_roti_3days_12_17utc.parquet"
        df_igs = pd.read_parquet(igs_pq) if igs_pq.exists() else pd.DataFrame()
        if not df_igs.empty:
            df_igs = df_igs[df_igs["date"] == f"{year}-{month:02d}-{day:02d}"].copy()

        om_path = precursors.fetch_omni2_year(year, ROOT / "data" / "raw" / "omni")
        df_om = precursors.load_omni2_hourly(om_path)
        d_obj = datetime(year, month, day).date()
        df_om = df_om[df_om.time.dt.date == d_obj].copy()
        inx_path = precursors.fetch_esa_ionex(d_obj, ROOT / "data" / "raw" / "ionex")
        df_inx = precursors.parse_ionex_eia_gradient(inx_path)
        return df_roti, df_c2, df_igs, df_om, df_inx

    sel = st.sidebar.selectbox("Chọn ngày đối chứng", list(DAYS.keys()))
    y, m, d, doy = DAYS[sel]
    df_roti, df_c2, df_igs, df_om, df_inx = load_day_bundle(y, m, d, doy)

    metric_2d = st.sidebar.radio("Biến nền 2D (Madrigal 3500)", ["roti", "tec"])
    show_sporadic_e = st.sidebar.checkbox("3D: Hiển thị thêm vùng thấp < 200 km (Sporadic-E)", value=False)

    times_5m = sorted(df_roti["time"].unique())
    cur_t = st.select_slider(
        "Khung thời gian tối VN (UTC)",
        options=times_5m,
        value=times_5m[min(24, len(times_5m) - 1)],
        format_func=lambda t: pd.Timestamp(t).strftime("%H:%M"),
    )
    cur_ts = pd.Timestamp(cur_t)
    cur_5m = cur_ts.floor("5min")
    st.caption(f"Thời điểm đang chọn: **{cur_ts.strftime('%Y-%m-%d %H:%M')} UTC** ({(cur_ts.hour + 7) % 24:02d}:{cur_ts.minute:02d} giờ VN)")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Bản đồ 2D Khu vực + Dip Equator + IGS + COSMIC-2")
        sub2d = df_roti[df_roti["time"] == cur_t]
        lats = np.arange(0, 29, 1)
        lons = np.arange(95, 123, 1)
        piv = sub2d.pivot(index="gdlat", columns="glon", values=metric_2d).reindex(index=lats, columns=lons)

        fig2d = go.Figure()
        fig2d.add_trace(go.Heatmap(
            z=piv.values, x=lons, y=lats,
            colorscale="YlOrRd" if metric_2d == "roti" else "Viridis",
            zmin=0, zmax=1.5 if metric_2d == "roti" else 120,
            colorbar=dict(title=metric_2d.upper()),
            hoverongaps=False,
        ))
        fig2d.add_trace(go.Scatter(x=[95, 122], y=[9.0, 9.0], mode="lines", name="Dip Equator (~9°N)",
                                   line=dict(color="cyan", dash="dash", width=2)))
        fig2d.add_trace(go.Scatter(x=[95, 122], y=[20.0, 20.0], mode="lines", name="Đỉnh EIA (~20°N)",
                                   line=dict(color="magenta", dash="dot", width=2)))
        fig2d.add_shape(type="rect", x0=101, y0=6, x1=116, y1=24, line=dict(color="lime", width=2))
        if not df_c2.empty:
            c2_win = df_c2[(df_c2.occheight_km >= 200.0) &
                           (df_c2["time"] >= cur_t - pd.Timedelta(minutes=15)) &
                           (df_c2["time"] <= cur_t + pd.Timedelta(minutes=15))]
            if not c2_win.empty:
                fig2d.add_trace(go.Scatter(
                    x=c2_win["lon"], y=c2_win["lat"], mode="markers", name="COSMIC-2 S4 (F>=200km)",
                    marker=dict(size=7, color=c2_win["s4_L1"], colorscale="Turbo", cmin=0, cmax=0.8, line=dict(width=0.5, color="black")),
                    text=[f"S4={v:.2f}, h={h:.0f}km" for v, h in zip(c2_win["s4_L1"], c2_win["occheight_km"])],
                ))
        if not df_igs.empty:
            igs_t = df_igs[df_igs["time_5min"] == cur_5m]
            if not igs_t.empty:
                fig2d.add_trace(go.Scatter(
                    x=igs_t["lon"], y=igs_t["lat"], mode="markers+text", name="IGS 30s ROTI",
                    text=[f"{s} ({v:.2f})" for s, v in zip(igs_t["station"], igs_t["igs_roti_p95"])],
                    textposition="top center",
                    marker=dict(size=12, symbol="diamond", color=igs_t["igs_roti_p95"], colorscale="Hot", cmin=0, cmax=1.5, line=dict(width=1, color="white")),
                ))
        fig2d.update_layout(
            xaxis_title="Kinh độ (°E)", yaxis_title="Vĩ độ (°N)", height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=10, r=10, t=55, b=10),
        )
        st.plotly_chart(fig2d, use_container_width=True)

    with col2:
        st.subheader("2. Bản đồ 3D Lai (Tầng F ≥ 200 km vs Sporadic-E < 200 km)")
        fig3d = go.Figure()
        fig3d.add_trace(go.Scatter3d(
            x=[101, 116, 116, 101, 101], y=[6, 6, 24, 24, 6], z=[0, 0, 0, 0, 0],
            mode="lines", name="Khung VN (z=0)", line=dict(color="green", width=4),
        ))
        fig3d.add_trace(go.Scatter3d(
            x=[95, 122], y=[9, 9], z=[0, 0], mode="lines", name="Dip Equator (z=0)",
            line=dict(color="cyan", width=3, dash="dash"),
        ))
        if not df_c2.empty:
            c2_f = df_c2[df_c2.occheight_km >= 200.0]
            fig3d.add_trace(go.Scatter3d(
                x=c2_f["lon"], y=c2_f["lat"], z=c2_f["occheight_km"], mode="markers",
                name="Tầng F (≥200 km)",
                marker=dict(size=3 + 8 * c2_f["s4_L1"].fillna(0), color=c2_f["s4_L1"], colorscale="YlOrRd", cmin=0, cmax=0.8),
            ))
            if show_sporadic_e:
                c2_e = df_c2[df_c2.occheight_km < 200.0]
                fig3d.add_trace(go.Scatter3d(
                    x=c2_e["lon"], y=c2_e["lat"], z=c2_e["occheight_km"], mode="markers",
                    name="Vùng thấp <200 km (Sporadic-E)",
                    marker=dict(size=3, color="gray", opacity=0.4),
                ))
        fig3d.update_layout(
            scene=dict(
                xaxis_title="Lon (°E)", yaxis_title="Lat (°N)", zaxis_title="Độ cao (km)",
                zaxis=dict(range=[0, 560]),
                aspectmode="manual", aspectratio=dict(x=1.2, y=1.2, z=0.8),
                camera=dict(eye=dict(x=1.55, y=-1.65, z=0.95)),
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            height=500, margin=dict(l=10, r=10, t=55, b=10),
        )
        st.plotly_chart(fig3d, use_container_width=True)

    st.subheader("3. Biểu đồ Keogram (12–17 UTC)")
    k1, k2 = st.columns(2)
    with k1:
        lon_t = df_roti.groupby(["glon", "time"])["roti"].quantile(0.95).unstack(fill_value=np.nan)
        fig_klon = go.Figure(go.Heatmap(
            z=lon_t.values, x=[pd.Timestamp(t).strftime("%H:%M") for t in lon_t.columns], y=lon_t.index,
            colorscale="YlOrRd", zmin=0, zmax=1.5, colorbar=dict(title="ROTI p95"),
        ))
        fig_klon.update_layout(title="Keogram Kinh độ – Thời gian (Trôi Đông)", xaxis_title="UTC", yaxis_title="Kinh độ (°E)", height=320)
        st.plotly_chart(fig_klon, use_container_width=True)
    with k2:
        lat_t = df_roti.groupby(["gdlat", "time"])["roti"].quantile(0.95).unstack(fill_value=np.nan)
        fig_klat = go.Figure(go.Heatmap(
            z=lat_t.values, x=[pd.Timestamp(t).strftime("%H:%M") for t in lat_t.columns], y=lat_t.index,
            colorscale="YlOrRd", zmin=0, zmax=1.5, colorbar=dict(title="ROTI p95"),
        ))
        fig_klat.add_hline(y=9.0, line_dash="dash", line_color="cyan")
        fig_klat.update_layout(title="Keogram Vĩ độ – Thời gian (Dip Equator → EIA)", xaxis_title="UTC", yaxis_title="Vĩ độ (°N)", height=320)
        st.plotly_chart(fig_klat, use_container_width=True)

    st.subheader("4. Tiền tố Vật lý (OMNI2 + ESA IONEX) & Bảng Cảnh báo 3 Miền")
    p_col, r_col = st.columns([2, 1])
    with p_col:
        aft = df_inx[(df_inx.utc_hour >= 7) & (df_inx.utc_hour <= 10)]
        st.dataframe(df_om[["time", "f107", "bz_gsm", "v_sw", "ey_mvm", "kp", "dst"]].tail(12), use_container_width=True)
        st.info(f"Độ chênh EIA buổi chiều (14–17h LT từ ESA IONEX): **Bắc={aft.tec_eia_north.mean():.1f}**, **Nam={aft.tec_trough_south.mean():.1f}**, **Diff={aft.eia_diff_ns.mean():.1f} TECU**")
    with r_col:
        c2_f = df_c2[df_c2.occheight_km >= 200.0] if not df_c2.empty else pd.DataFrame()
        for name, lo, hi in [("Bắc Bộ (18–24°N)", 18, 24), ("Trung Bộ (12–18°N)", 12, 18), ("Nam Bộ (6–12°N)", 6, 12)]:
            sub_c2 = c2_f[c2_f.lat.between(lo, hi) & c2_f.lon.between(101, 116)] if not c2_f.empty else pd.DataFrame()
            s4_p95 = float(sub_c2["s4_L1"].quantile(0.95)) if not sub_c2.empty else float("nan")
            if np.isnan(s4_p95):
                st.info(f"**{name}**: Không có lát cắt COSMIC-2")
            elif s4_p95 >= 0.5:
                st.error(f"**{name}** — ĐỎ (Nguy cơ mất khóa RTK) | `S4 p95 = {s4_p95:.2f}`")
            elif s4_p95 >= 0.25:
                st.warning(f"**{name}** — VÀNG (Nhiễu vừa) | `S4 p95 = {s4_p95:.2f}`")
            else:
                st.success(f"**{name}** — XANH (Bình thường) | `S4 p95 = {s4_p95:.2f}`")
