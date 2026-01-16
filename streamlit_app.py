import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# Konfigurasi Halaman
st.set_page_config(
    page_title="Price Level Predictor",
    page_icon="📈",
    layout="wide"
)

# Judul Aplikasi
st.title("📈 Dashboard Prediksi Tingkat Harga Komoditas")
st.markdown("""
Aplikasi ini memprediksi tingkat harga komoditas (Rendah, Sedang, Tinggi) berdasarkan data historis, 
kurs mata uang, dan tren pasar global.
""")

# Fungsi untuk memuat artefak
@st.cache_resource
def load_artifacts():
    path = 'artifacts/'
    artifacts = {
        'model': joblib.load(os.path.join(path, 'best_model.pkl')),
        'scaler': joblib.load(os.path.join(path, 'scaler.pkl')),
        'le_komoditas': joblib.load(os.path.join(path, 'komoditas_encoder.pkl')),
        'le_provinsi': joblib.load(os.path.join(path, 'provinsi_encoder.pkl')),
        'target_encoder': joblib.load(os.path.join(path, 'target_encoder.pkl')),
        'feature_list': joblib.load(os.path.join(path, 'feature_list.pkl'))
    }
    return artifacts

try:
    art = load_artifacts()
    st.sidebar.success("✅ Model & Artefak berhasil dimuat")
except Exception as e:
    st.error(f"❌ Gagal memuat artefak: {e}")
    st.stop()

# --- SIDEBAR: INPUT USER ---
st.sidebar.header("Input Parameter")

# 1. Input Kategori
selected_komoditas = st.sidebar.selectbox("Pilih Komoditas", art['le_komoditas'].classes_)
selected_provinsi = st.sidebar.selectbox("Pilih Provinsi", art['le_provinsi'].classes_)

# 2. Input Tanggal (untuk ekstraksi fitur waktu)
selected_date = st.sidebar.date_input("Pilih Tanggal Prediksi", datetime.now())

# 3. Input Numerik (Harga & Global)
st.sidebar.subheader("Data Pasar & Global")

# Membagi kolom agar rapi
col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Harga Lokal")
    harga_lag_1 = st.number_input("Harga H-1 (IDR)", value=15000.0)
    harga_lag_7 = st.number_input("Harga H-7 (IDR)", value=14800.0)
    harga_lag_30 = st.number_input("Harga H-30 (IDR)", value=14500.0)
    google_trend = st.number_input("Skor Google Trend", value=50.0)

    st.subheader("Kurs Mata Uang")
    kurs_usdidr = st.number_input("USD/IDR", value=15600.0)
    kurs_myrusd = st.number_input("MYR/USD", value=0.21)
    kurs_sgdusd = st.number_input("SGD/USD", value=0.74)
    kurs_thbusd = st.number_input("THB/USD", value=0.028)

with col2:
    st.subheader("Komoditas Global")
    global_crude_oil = st.number_input("Crude Oil (USD)", value=80.0)
    global_natural_gas = st.number_input("Natural Gas (USD)", value=2.5)
    global_coal = st.number_input("Coal (USD)", value=130.0)
    global_palm_oil = st.number_input("Palm Oil (USD)", value=900.0)
    global_sugar = st.number_input("Sugar (USD)", value=0.20)
    global_wheat = st.number_input("Wheat (USD)", value=600.0)

# --- PROSES PREDIKSI ---
if st.button("Hitung Prediksi"):
    # 1. Ekstraksi Fitur Waktu
    month = selected_date.month
    day = selected_date.day
    week = selected_date.isocalendar()[1]
    day_of_week = selected_date.weekday()

    # 2. Encoding Kategorikal
    encoded_komo = art['le_komoditas'].transform([selected_komoditas])[0]
    encoded_prov = art['le_provinsi'].transform([selected_provinsi])[0]

    # 3. Scaling Numerikal
    # Ambil urutan fitur numerik dari scaler (14 fitur)
    numerical_input = np.array([[
        harga_lag_1, harga_lag_7, harga_lag_30, 
        kurs_myrusd, kurs_sgdusd, kurs_thbusd, kurs_usdidr,
        global_crude_oil, global_natural_gas, global_coal, 
        global_palm_oil, global_sugar, global_wheat, google_trend
    ]])
    scaled_numerical = art['scaler'].transform(numerical_input)[0]

    # 4. Gabungkan Semua Fitur Sesuai Urutan di feature_list.pkl
    # Urutan: komoditas, provinsi, month, week, day, day_of_week, [14 numerik]
    input_data = [encoded_komo, encoded_prov, month, week, day, day_of_week] + list(scaled_numerical)
    
    # 5. Prediksi
    prediction_idx = art['model'].predict([input_data])[0]
    prediction_label = art['target_encoder.pkl'.replace('.pkl', '')].classes_[prediction_idx] if hasattr(art['target_encoder'], 'classes_') else art['target_encoder'].inverse_transform([prediction_idx])[0]
    
    # Menampilkan Hasil
    st.divider()
    st.subheader("Hasil Analisis")
    
    color = "red" if prediction_label == "Tinggi" else "orange" if prediction_label == "Sedang" else "green"
    
    st.markdown(f"""
    <div style="padding:20px; border-radius:10px; background-color:#f0f2f6; border-left: 10px solid {color};">
        <h2 style="margin:0;">Tingkat Harga: <span style="color:{color};">{prediction_label}</span></h2>
        <p style="margin:10px 0 0 0;">Prediksi ini untuk <b>{selected_komoditas}</b> di wilayah <b>{selected_provinsi}</b> 
        pada tanggal {selected_date.strftime('%d %B %Y')}.</p>
    </div>
    """, unsafe_allow_html=True)

    # Probabilitas (Jika model mendukung)
    try:
        probs = art['model'].predict_proba([input_data])[0]
        prob_df = pd.DataFrame({
            'Kategori': art['target_encoder'].classes_,
            'Keyakinan (%)': [p * 100 for p in probs]
        })
        st.write("### Probabilitas Prediksi")
        st.bar_chart(prob_df.set_index('Kategori'))
    except:
        pass

# Footer
st.markdown("---")
st.caption("Deployment by Your Name | Model by Teammate")
