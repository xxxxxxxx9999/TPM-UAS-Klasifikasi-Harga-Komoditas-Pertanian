import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# =================================================================
# 1. KONFIGURASI HALAMAN & STYLE
# =================================================================
st.set_page_config(
    page_title="Price Intel Indonesia",
    page_icon="🏪",
    layout="wide"
)

# Custom CSS untuk UI Premium
st.markdown("""
    <style>
    .main-card {
        padding: 30px;
        border-radius: 20px;
        background-color: #ffffff;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        border: 1px solid #f0f2f6;
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
    }
    .result-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 5px;
        text-align: center;
    }
    .result-value {
        font-size: 4rem;
        font-weight: 900;
        text-align: center;
        margin: 0;
        line-height: 1;
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
# 3. HEADER & NAVIGASI
# =================================================================
st.title("🏪 Price Intel: Prediksi Harga Komoditas")
st.markdown("Analisis cerdas untuk memprediksi stabilitas harga pangan di Indonesia.")

# Pemilihan Mode
mode = st.radio("Pilih Mode Input:", ["Standard (Cepat & Lokal)", "Advanced (Full Parameter)"], horizontal=True)

st.divider()

# =================================================================
# 4. FORM INPUT BERDASARKAN MODE
# =================================================================

# Nilai Default untuk Mode Standard (Latar Belakang)
defaults = {
    'kurs_usdidr': 15600.0, 'kurs_myrusd': 0.21, 'kurs_sgdusd': 0.74, 'kurs_thbusd': 0.028,
    'g_oil': 80.0, 'g_gas': 2.5, 'g_coal': 130.0, 'g_palm': 900.0, 'g_sugar': 0.20, 'g_wheat': 600.0,
    'trend': 50.0
}

with st.container():
    col_main1, col_main2 = st.columns([1, 2])
    
    with col_main1:
        st.subheader("📍 Data Utama")
        sel_komo = st.selectbox("Komoditas", assets['le_komo'].classes_)
        sel_prov = st.selectbox("Provinsi", assets['le_prov'].classes_)
        sel_date = st.date_input("Tanggal Prediksi", datetime.now())
        
        st.subheader("💰 Harga Historis")
        h_l1 = st.number_input("Harga Kemarin (H-1)", value=15000.0, step=500.0)
        
        if mode == "Standard (Cepat & Lokal)":
            # Mode Sederhana: Lag 7 dan 30 otomatis mendekati Lag 1 jika tidak diisi
            h_l7 = h_l1 - 200.0
            h_l30 = h_l1 - 500.0
            st.caption("ℹ️ Mode Standar menggunakan estimasi otomatis untuk data global & kurs.")
        else:
            h_l7 = st.number_input("Harga Minggu Lalu (H-7)", value=14800.0)
            h_l30 = st.number_input("Harga Bulan Lalu (H-30)", value=14500.0)

    with col_main2:
        if mode == "Advanced (Full Parameter)":
            st.subheader("🌐 Indikator Ekonomi & Global")
            c1, c2 = st.columns(2)
            with c1:
                k_usdidr = st.number_input("USD/IDR", value=defaults['kurs_usdidr'])
                k_myr = st.number_input("MYR/USD", value=defaults['kurs_myrusd'], format="%.4f")
                k_sgd = st.number_input("SGD/USD", value=defaults['kurs_sgdusd'], format="%.4f")
                k_thb = st.number_input("THB/USD", value=defaults['kurs_thbusd'], format="%.4f")
                g_trend = st.number_input("Google Trend", value=defaults['trend'])
            with c2:
                g_oil = st.number_input("Crude Oil", value=defaults['g_oil'])
                g_coal = st.number_input("Coal", value=defaults['g_coal'])
                g_palm = st.number_input("CPO (Minyak Sawit)", value=defaults['g_palm'])
                g_wheat = st.number_input("Gandum", value=defaults['g_wheat'])
        else:
            # Menggunakan nilai default jika di mode standard
            k_usdidr, k_myr, k_sgd, k_thb = defaults['kurs_usdidr'], defaults['kurs_myrusd'], defaults['kurs_sgdusd'], defaults['kurs_thbusd']
            g_oil, g_coal, g_palm, g_wheat, g_trend = defaults['g_oil'], defaults['g_coal'], defaults['g_palm'], defaults['g_wheat'], defaults['trend']
            
            # Tampilan visual pengganti di mode standard
            st.info("💡 **Mode Standard Aktif**\n\nSistem akan memproses prediksi menggunakan variabel harga lokal yang Anda masukkan dan memadukannya dengan rata-rata indikator pasar global saat ini.")
            st.image("https://img.freemarket.com/vectors/market-analysis-concept_23-2148560000.jpg", use_container_width=True, caption="Analisis Pasar Otomatis")

# =================================================================
# 5. EKSEKUSI PREDIKSI
# =================================================================
if st.button("CEK PREDIKSI HARGA", type="primary"):
    # 1. Waktu
    m, d, w, dow = sel_date.month, sel_date.day, sel_date.isocalendar()[1], sel_date.weekday()
    
    # 2. Encode
    e_komo = assets['le_komo'].transform([sel_komo])[0]
    e_prov = assets['le_prov'].transform([sel_prov])[0]
    
    # 3. Scale Numerik (Harus 14 Fitur)
    # Urutan sesuai scaler: h_l1, h_l7, h_l30, k_myr, k_sgd, k_thb, k_usdidr, g_oil, gas, coal, palm, sugar, wheat, trend
    # Catatan: g_gas dan g_sugar kita pakai default karena jarang berubah drastis
    num_input = [h_l1, h_l7, h_l30, k_myr, k_sgd, k_thb, k_usdidr, g_oil, defaults['g_gas'], g_coal, g_palm, defaults['g_sugar'], g_wheat, g_trend]
    scaled_num = assets['scaler'].transform([num_input])[0]
    
    # 4. Final Input (20 fitur)
    final_input = [e_komo, e_prov, m, w, d, dow] + list(scaled_num)
    
    # 5. Predict
    res_idx = assets['model'].predict([final_input])[0]
    label = assets['target_enc'].classes_[res_idx]
    
    # 6. Tampilan Hasil UI Pro
    colors = {"Tinggi": ("#FF4B4B", "#FFF5F5"), "Sedang": ("#FFA500", "#FFF9EE"), "Rendah": ("#28A745", "#F2FFF5")}
    main_color, bg_color = colors.get(label, ("#333", "#eee"))

    st.markdown(f"""
        <div class="main-card" style="background-color: {bg_color}; border-top: 10px solid {main_color};">
            <p class="result-header">Estimasi Tingkat Harga untuk {sel_komo}</p>
            <p class="result-value" style="color: {main_color};">{label.upper()}</p>
            <p style="text-align: center; color: #555; margin-top: 15px;">
                Wilayah: <b>{sel_prov}</b> | Prediksi Tanggal: <b>{sel_date.strftime('%d %B %Y')}</b>
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Chart Probabilitas
    try:
        probs = assets['model'].predict_proba([final_input])[0]
        with st.expander("Lihat Detail Probabilitas"):
            p_df = pd.DataFrame({'Level': assets['target_enc'].classes_, 'Confidence (%)': probs*100})
            st.bar_chart(p_df.set_index('Level'))
    except:
        pass

st.markdown("---")
st.caption
