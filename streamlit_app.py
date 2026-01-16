import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# =================================================================
# 1. KONFIGURASI HALAMAN
# =================================================================
st.set_page_config(
    page_title="Prediksi Tingkat Harga Komoditas",
    page_icon="📈",
    layout="wide"
)

# Custom CSS untuk memastikan teks di kotak hasil selalu terlihat
st.markdown("""
    <style>
    .result-box {
        padding: 40px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .result-label {
        font-size: 50px;
        font-weight: bold;
        margin: 0;
        text-transform: uppercase;
    }
    .result-text {
        color: #333333 !important; /* Memaksa warna teks gelap agar kontras */
        font-family: sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# =================================================================
# 2. LOAD ARTEFAK MODEL (CACHED)
# =================================================================
@st.cache_resource
def load_artifacts():
    path = 'artifacts/'
    try:
        return {
            'model': joblib.load(os.path.join(path, 'best_model.pkl')),
            'scaler': joblib.load(os.path.join(path, 'scaler.pkl')),
            'le_komoditas': joblib.load(os.path.join(path, 'komoditas_encoder.pkl')),
            'le_provinsi': joblib.load(os.path.join(path, 'provinsi_encoder.pkl')),
            'target_encoder': joblib.load(os.path.join(path, 'target_encoder.pkl')),
            'features': joblib.load(os.path.join(path, 'feature_list.pkl'))
        }
    except Exception as e:
        st.error(f"Gagal memuat artefak: {e}")
        return None

artifacts = load_artifacts()

if not artifacts:
    st.stop()

# =================================================================
# 3. ANTARMUKA PENGGUNA (UI)
# =================================================================
st.title("📈 Dashboard Prediksi Harga Komoditas")
st.info("Sistem ini memprediksi level harga (Rendah, Sedang, Tinggi)")

# --- SIDEBAR UNTUK INPUT ---
st.sidebar.header("⚙️ Konfigurasi Input")

# Kategori
selected_komo = st.sidebar.selectbox("Jenis Komoditas", artifacts['le_komoditas'].classes_)
selected_prov = st.sidebar.selectbox("Wilayah Provinsi", artifacts['le_provinsi'].classes_)
selected_date = st.sidebar.date_input("Tanggal Prediksi", datetime.now())

# Parameter Numerik (Membagi kolom agar hemat ruang)
st.sidebar.subheader("📊 Data Indikator")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Data Lokal")
    h_lag_1 = st.number_input("Harga Kemarin (H-1)", value=15000)
    h_lag_7 = st.number_input("Harga H-7", value=14800)
    h_lag_30 = st.number_input("Harga H-30", value=14500)
    g_trend = st.number_input("Google Trend (0-100)", value=50.0)
    
    st.subheader("💹 Kurs Valuta")
    k_usdidr = st.number_input("USD to IDR", value=15600)
    k_myr = st.number_input("MYR to USD", value=0.21, format="%.4f")
    k_sgd = st.number_input("SGD to USD", value=0.74, format="%.4f")
    k_thb = st.number_input("THB to USD", value=0.028, format="%.4f")

with col2:
    st.subheader("🌎 Indikator Global")
    g_oil = st.number_input("Minyak Mentah (Crude Oil)", value=80.0)
    g_gas = st.number_input("Gas Alam", value=2.5)
    g_coal = st.number_input("Batu Bara (Coal)", value=130.0)
    g_palm = st.number_input("Minyak Sawit (CPO)", value=900.0)
    g_sugar = st.number_input("Gula Dunia", value=0.20, format="%.4f")
    g_wheat = st.number_input("Gandum Dunia", value=600.0)

# =================================================================
# 4. LOGIKA PREDIKSI
# =================================================================
if st.button("🚀 JALANKAN PREDIKSI", type="primary", use_container_width=True):
    
    # A. Preprocessing: Fitur Waktu
    month = selected_date.month
    day = selected_date.day
    week = selected_date.isocalendar()[1]
    day_of_week = selected_date.weekday()

    # B. Preprocessing: Encoding Kategorikal
    enc_komo = artifacts['le_komoditas'].transform([selected_komo])[0]
    enc_prov = artifacts['le_provinsi'].transform([selected_prov])[0]

    # C. Preprocessing: Scaling Numerikal (Urutan harus sesuai scaler)
    num_vals = [
        h_lag_1, h_lag_7, h_lag_30, 
        k_myr, k_sgd, k_thb, k_usdidr,
        g_oil, g_gas, g_coal, g_palm, g_sugar, g_wheat, g_trend
    ]
    scaled_vals = artifacts['scaler'].transform([num_vals])[0]

    # D. Gabungkan Menjadi 20 Fitur Sesuai feature_list.pkl
    # Urutan: [komoditas, provinsi, month, week, day, day_of_week] + [14 numerik]
    input_final = [enc_komo, enc_prov, month, week, day, day_of_week] + list(scaled_vals)
    
    # E. Jalankan Prediksi
    pred_idx = artifacts['model'].predict([input_final])[0]
    label = artifacts['target_encoder'].classes_[pred_idx]

    # F. Visualisasi Hasil
    st.markdown("---")
    
    # Logika Warna
    if label == "Tinggi":
        bg, border, text_color = "#fef2f2", "#dc2626", "#dc2626"
    elif label == "Sedang":
        bg, border, text_color = "#fffbeb", "#d97706", "#d97706"
    else:
        bg, border, text_color = "#f0fdf4", "#16a34a", "#16a34a"

    # Menampilkan Kotak Hasil
    st.markdown(f"""
        <div class="result-box" style="background-color: {bg}; border: 4px solid {border};">
            <p class="result-text" style="font-size: 20px; margin: 0;">Prediksi Tingkat Harga:</p>
            <h1 class="result-label" style="color: {text_color};">{label}</h1>
            <div style="height: 2px; background: {border}; width: 100px; margin: 15px auto;"></div>
            <p class="result-text" style="font-size: 16px;">
                <b>{selected_komo}</b> di <b>{selected_prov}</b><br>
                Target: {selected_date.strftime('%d %B %Y')}
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Probabilitas Chart
    try:
        probs = artifacts['model'].predict_proba([input_final])[0]
        st.subheader("📊 Analisis Probabilitas")
        prob_df = pd.DataFrame({
            'Level': artifacts['target_encoder'].classes_,
            'Keyakinan (%)': [p * 100 for p in probs]
        })
        st.bar_chart(prob_df.set_index('Level'))
    except:
        pass

# Footer
st.markdown("---")
st.caption("Deployment Selesai | Dikembangkan untuk Analisis Harga Komoditas Indonesia")

