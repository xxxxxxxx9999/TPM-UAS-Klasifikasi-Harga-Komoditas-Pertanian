import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
from datetime import datetime, timedelta

# =================================================================
# 1. KONFIGURASI HALAMAN & STYLE
# =================================================================
st.set_page_config(
    page_title="Nama Projek",
    page_icon="📈",
    layout="wide"
)

# Custom CSS untuk UI Modern & Elegan
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-card {
        padding: 40px;
        border-radius: 24px;
        background-color: #ffffff;
        box-shadow: 0 20px 40px rgba(0,0,0,0.05);
        border: 1px solid #f0f2f6;
        margin-bottom: 30px;
        text-align: center;
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 4em;
        font-weight: 700;
        background: linear-gradient(135deg, #FF4B2B 0%, #FF416C 100%);
        color: white;
        border: none;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(255, 75, 43, 0.3);
    }

    .stat-label {
        font-size: 1rem;
        color: #888;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .prediction-label {
        font-size: 5rem;
        font-weight: 800;
        margin: 10px 0;
        line-height: 1;
    }
    
    .info-text {
        color: #555;
        font-size: 1.1rem;
    }
    
    /* Menghilangkan border pada expander */
    .streamlit-expanderHeader {
        border: none !important;
        background-color: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)

# =================================================================
# 2. LOAD ARTEFAK
# =================================================================
@st.cache_resource
def load_model_assets():
    path = 'artifacts/'
    try:
        return {
            'model': joblib.load(os.path.join(path, 'best_model.pkl')),
            'scaler': joblib.load(os.path.join(path, 'scaler.pkl')),
            'le_komo': joblib.load(os.path.join(path, 'komoditas_encoder.pkl')),
            'le_prov': joblib.load(os.path.join(path, 'provinsi_encoder.pkl')),
            'target_enc': joblib.load(os.path.join(path, 'target_encoder.pkl')),
            'feat_list': joblib.load(os.path.join(path, 'feature_list.pkl'))
        }
    except Exception as e:
        st.error(f"Gagal memuat sistem: {e}")
        return None

assets = load_model_assets()
if not assets: st.stop()

# =================================================================
# 3. HEADER
# =================================================================
st.title("📈 Nama Projek")
st.write("Analisis prediktif untuk stabilitas harga pangan nasional.")

# Pemilihan Mode dengan Desain Minimalis
mode = st.select_slider(
    "Pilih Mode Analisis",
    options=["Standard (Cepat)", "Advanced (Lengkap)"],
    help="Gunakan Advanced jika Anda ingin memasukkan data ekonomi global secara manual."
)

st.divider()

# =================================================================
# 4. FORM INPUT
# =================================================================
defaults = {
    'kurs_usdidr': 15600.0, 'kurs_myrusd': 0.21, 'kurs_sgdusd': 0.74, 'kurs_thbusd': 0.028,
    'g_oil': 80.0, 'g_gas': 2.5, 'g_coal': 130.0, 'g_palm': 900.0, 'g_sugar': 0.20, 'g_wheat': 600.0,
    'trend': 50.0
}

with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📍 Detail Wilayah & Komoditas")
        sel_komo = st.selectbox("Komoditas", assets['le_komo'].classes_)
        sel_prov = st.selectbox("Provinsi", assets['le_prov'].classes_)
        sel_date = st.date_input("Target Tanggal Prediksi", datetime.now() + timedelta(days=1))
        
    with col2:
        st.markdown("### 💰 Data Harga Lokal")
        h_l1 = st.number_input("Harga Hari Ini (H-1)", value=15000.0, step=500.0)
        
        if mode == "Standard (Cepat)":
            h_l7 = h_l1 * 0.98  # Estimasi otomatis
            h_l30 = h_l1 * 0.95 # Estimasi otomatis
            st.info("💡 Mode standar menggunakan estimasi cerdas untuk data kurs & pasar global.")
        else:
            h_l7 = st.number_input("Harga Minggu Lalu (H-7)", value=14800.0)
            h_l30 = st.number_input("Harga Bulan Lalu (H-30)", value=14500.0)

    if mode == "Advanced (Lengkap)":
        st.divider()
        st.markdown("### 🌐 Indikator Global & Ekonomi")
        adv_col1, adv_col2 = st.columns(2)
        with adv_col1:
            k_usdidr = st.number_input("Kurs USD/IDR", value=defaults['kurs_usdidr'])
            k_myr = st.number_input("Kurs MYR/USD", value=defaults['kurs_myrusd'], format="%.4f")
            g_trend = st.number_input("Google Trend Score", value=defaults['trend'])
        with adv_col2:
            g_oil = st.number_input("Harga Minyak Dunia (Crude Oil)", value=defaults['g_oil'])
            g_palm = st.number_input("Harga Palm Oil (CPO)", value=defaults['g_palm'])
            g_wheat = st.number_input("Harga Gandum Dunia", value=defaults['g_wheat'])
        k_sgd, k_thb = defaults['kurs_sgdusd'], defaults['kurs_thbusd']
    else:
        k_usdidr, k_myr, k_sgd, k_thb = defaults['kurs_usdidr'], defaults['kurs_myrusd'], defaults['kurs_sgdusd'], defaults['kurs_thbusd']
        g_oil, g_coal, g_palm, g_wheat, g_trend = defaults['g_oil'], defaults['g_coal'], defaults['g_palm'], defaults['g_wheat'], defaults['trend']

# =================================================================
# 5. PREDIKSI & VISUALISASI MODERN
# =================================================================
st.markdown("<br>", unsafe_allow_html=True)
if st.button("MULAI ANALISIS PREDIKSI"):
    # Preprocessing
    m, d, w, dow = sel_date.month, sel_date.day, sel_date.isocalendar()[1], sel_date.weekday()
    e_komo = assets['le_komo'].transform([sel_komo])[0]
    e_prov = assets['le_prov'].transform([sel_prov])[0]
    
    num_input = [h_l1, h_l7, h_l30, k_myr, k_sgd, k_thb, k_usdidr, g_oil, defaults['g_gas'], defaults['g_coal'], g_palm, defaults['g_sugar'], g_wheat, g_trend]
    scaled_num = assets['scaler'].transform([num_input])[0]
    final_input = [e_komo, e_prov, m, w, d, dow] + list(scaled_num)
    
    # Model Predict
    res_idx = assets['model'].predict([final_input])[0]
    label = assets['target_enc'].classes_[res_idx]
    
    # UI Logic
    ui_map = {
        "Tinggi": ("#FF4B2B", "#FFF5F5"),
        "Sedang": ("#F9D423", "#FFFDF0"),
        "Rendah": ("#00B09B", "#F0FFF9")
    }
    main_color, bg_color = ui_map.get(label, ("#333", "#eee"))

    # Output Card
    st.markdown(f"""
        <div class="main-card" style="background-color: {bg_color}; border-top: 12px solid {main_color};">
            <p class="stat-label">Hasil Prediksi untuk {sel_komo}</p>
            <p class="prediction-label" style="color: {main_color};">{label.upper()}</p>
            <div style="height: 2px; background: linear-gradient(to right, transparent, {main_color}, transparent); width: 60%; margin: 20px auto;"></div>
            <p class="info-text">
                Estimasi tingkat harga di wilayah <b>{sel_prov}</b><br>
                pada tanggal <b>{sel_date.strftime('%d %B %Y')}</b>
            </p>
        </div>
    """, unsafe_allow_html=True)

    # MODERN CHART: Plotly Line Chart
    st.markdown("### 📊 Visualisasi Tren & Keyakinan")
    
    # Layout untuk grafik
    chart_col1, chart_col2 = st.columns([2, 1])
    
    with chart_col1:
        # Data untuk grafik tren harga lokal yang diinput
        dates = [
            (datetime.now() - timedelta(days=30)).strftime('%d %b'),
            (datetime.now() - timedelta(days=7)).strftime('%d %b'),
            datetime.now().strftime('%d %b'),
            sel_date.strftime('%d %b') + " (Prediksi)"
        ]
        prices = [h_l30, h_l7, h_l1, h_l1] # Prediksi divisualkan sejajar dengan H-1 untuk konteks level
        
        fig = go.Figure()
        # Garis Histori
        fig.add_trace(go.Scatter(
            x=dates[:3], y=prices[:3],
            mode='lines+markers',
            name='Histori',
            line=dict(color='#333', width=4),
            marker=dict(size=10, color='#333')
        ))
        # Titik Prediksi
        fig.add_trace(go.Scatter(
            x=[dates[3]], y=[prices[3]],
            mode='markers',
            name='Target',
            marker=dict(size=18, color=main_color, symbol='star')
        ))
        
        fig.update_layout(
            title="Alur Harga Lokal & Target Prediksi",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=40, b=0),
            height=300,
            showlegend=False,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#eee')
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        # Bar Chart Keyakinan Model
        try:
            probs = assets['model'].predict_proba([final_input])[0]
            prob_fig = go.Figure(go.Bar(
                x=[p * 100 for p in probs],
                y=assets['target_enc'].classes_,
                orientation='h',
                marker_color=[ui_map[c][0] for c in assets['target_enc'].classes_]
            ))
            prob_fig.update_layout(
                title="Keyakinan Model (%)",
                height=300,
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(range=[0, 100])
            )
            st.plotly_chart(prob_fig, use_container_width=True)
        except:
            st.warning("Data probabilitas tidak tersedia.")

st.divider()
st.markdown("<p style='text-align: center; color: #bbb; font-size: 0.85rem; font-weight: 600;'>© 2024 Nama Projek • Smart Price Analytics</p>", unsafe_allow_html=True)
