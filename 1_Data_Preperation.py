# %%
# TAHAP 1 — AKUISISI DAN PEMUATAN DATA
# Tujuan: memuat data mentah secara valid dari direktori proyek
# Tahap 1.1 - Import library yang dibutuhkan
import pandas as pd
import numpy as np
from pathlib import Path


# Tahap 1.2 - Menentukan direktori proyek dan dataset
BASE_DIR = Path(__file__).resolve().parent
DATA_PERTANIAN_DIR = BASE_DIR / "Data_Pertanian"
HARGA_DIR = DATA_PERTANIAN_DIR / "Harga Bahan Pangan"

print("Direktori proyek:", BASE_DIR)
print("Folder Harga Bahan Pangan tersedia:", HARGA_DIR.exists())


# Tahap 1.3 - Pemeriksaan struktur folder Data_Pertanian
for p in DATA_PERTANIAN_DIR.iterdir():
    if p.is_dir():
        print("DIR :", p.name)
    else:
        print("FILE:", p.name)


# Tahap 1.4 - Menentukan folder data valid (train dan test)
DATA_FOLDERS = []

for nama_folder in ["train", "test"]:
    folder_path = HARGA_DIR / nama_folder
    if folder_path.exists():
        DATA_FOLDERS.append(folder_path)

print("Folder data yang digunakan:")
for d in DATA_FOLDERS:
    print("-", d)


# Tahap 1.5 - Memuat, menggabungkan data, dan standarisasi awal
csv_files = []
for folder in DATA_FOLDERS:
    csv_files.extend(folder.glob("*.csv"))

daftar_df = []

for file in csv_files:
    df_tmp = pd.read_csv(file)

    # Konversi Date ke datetime (WAJIB)
    df_tmp["Date"] = pd.to_datetime(df_tmp["Date"], errors="coerce")

    # Metadata
    df_tmp["commodity"] = file.stem
    df_tmp["source_folder"] = file.parent.name

    daftar_df.append(df_tmp)

df_raw = pd.concat(daftar_df, ignore_index=True)

print("Ukuran dataset mentah:", df_raw.shape)
df_raw.head()


# Tahap 1.6 - Quality check awal (missing value diagnostik)
# Ubah wide → long SEMENTARA hanya untuk QC
df_qc = (
    df_raw
    .melt(
        id_vars=["Date", "commodity"],
        value_vars=[c for c in df_raw.columns if c not in ["Date", "commodity", "source_folder"]],
        var_name="province",
        value_name="price"
    )
)

# Missing rate overall
missing_rate = df_qc["price"].isna().mean()
print("Missing rate (overall):", round(missing_rate * 100, 2), "%")

# Missing rate per komoditas
missing_by_commodity = (
    df_qc
    .assign(is_missing=df_qc["price"].isna())
    .groupby("commodity")["is_missing"]
    .mean()
    .sort_values(ascending=False)
)

print("\nMissing rate by commodity:")
print((missing_by_commodity * 100).round(2))





# %%
# TAHAP 2 — PEMAHAMAN AWAL DATA
# Tujuan: memahami struktur, skema, dan karakteristik dataset
# Tahap 2.1 - Inspeksi skema dan tipe data
print("Daftar kolom pada dataset:")
for col in df_raw.columns:
    print("-", col)

print("\nUkuran dataset:", df_raw.shape)
df_raw.info()


# Tahap 2.2 - Statistik deskriptif awal
df_raw.describe(include="all")


# Tahap 2.3 - Pemeriksaan rentang waktu dan identifikasi skema data
print("Tanggal minimum:", df_raw["Date"].min())
print("Tanggal maksimum:", df_raw["Date"].max())

kolom_tetap = ["Date", "commodity", "source_folder"]
kolom_provinsi = [c for c in df_raw.columns if c not in kolom_tetap]

print("\nKolom tetap:", kolom_tetap)
print("Jumlah kolom provinsi:", len(kolom_provinsi))





# %%
# TAHAP 3 — STANDARISASI SKEMA DATA (WIDE → LONG)
# Tujuan: menyatukan format data agar konsisten dan siap diproses
# Tahap 3.1 - Identifikasi kolom tetap dan kolom provinsi
kolom_tanggal = "Date"
kolom_komoditas = "commodity"   # nama kolom asli di df_raw

kolom_tetap = [kolom_tanggal, kolom_komoditas, "source_folder"]
kolom_provinsi = [c for c in df_raw.columns if c not in kolom_tetap]

print("Contoh kolom provinsi:", kolom_provinsi[:5])


# Tahap 3.2 - Transformasi data wide menjadi long format
df_long = (
    df_raw
    .melt(
        id_vars=[kolom_tanggal, kolom_komoditas],
        value_vars=kolom_provinsi,
        var_name="provinsi",
        value_name="harga"
    )
    .rename(columns={
        kolom_tanggal: "tanggal",
        kolom_komoditas: "komoditas"
    })
)

print("Ukuran data setelah transformasi:", df_long.shape)
df_long.head()


# Tahap 3.3 - Pemeriksaan kualitas data awal (pra-pembersihan)
print("Tipe data setiap kolom:")
print(df_long.dtypes)

print("\nRentang tanggal:",
      df_long["tanggal"].min(), "→", df_long["tanggal"].max())

jumlah_duplikasi = df_long.duplicated(
    subset=["tanggal", "komoditas", "provinsi"]
).sum()
print("\nJumlah baris duplikat:", jumlah_duplikasi)

tingkat_missing = df_long["harga"].isna().mean()
print("Persentase missing value (%):", round(tingkat_missing * 100, 2))




# %%
# TAHAP 4 — PEMBERSIHAN DATA DASAR
# Tujuan: menghilangkan noise utama tanpa merusak struktur waktu
# Tahap 4.1 - Kondisi data sebelum pembersihan
print("Ukuran data sebelum pembersihan:", df_long.shape)

print("\nMissing value per kolom (sebelum):")
print(df_long.isna().sum())

print("\nJumlah duplikasi (sebelum):",
      df_long.duplicated(subset=["tanggal", "komoditas", "provinsi"]).sum())


# Tahap 4.2 - Menghapus missing value pada kolom kritis
before = df_long.shape[0]

df_long = df_long.dropna(
    subset=["tanggal", "komoditas", "provinsi", "harga"]
)

after = df_long.shape[0]

print("Baris sebelum drop missing :", before)
print("Baris setelah drop missing :", after)
print("Jumlah baris dihapus       :", before - after)


# Tahap 4.3 - Menghapus harga tidak valid (≤ 0)
before = df_long.shape[0]

df_long = df_long[df_long["harga"] > 0]

after = df_long.shape[0]

print("Baris sebelum filter harga :", before)
print("Baris setelah filter harga :", after)
print("Jumlah baris dihapus       :", before - after)


# Tahap 4.4 - Menghapus data duplikat
before = df_long.shape[0]

df_long = df_long.drop_duplicates(
    subset=["tanggal", "komoditas", "provinsi"]
)

after = df_long.shape[0]

print("Baris sebelum deduplikasi :", before)
print("Baris setelah deduplikasi :", after)
print("Jumlah baris dihapus      :", before - after)


# Tahap 4.5 - Ringkasan data setelah pembersihan
print("Ukuran data setelah pembersihan:", df_long.shape)

print("\nMissing value per kolom (sesudah):")
print(df_long.isna().sum())

print("\nJumlah duplikasi (sesudah):",
      df_long.duplicated(subset=["tanggal", "komoditas", "provinsi"]).sum())

# %%
# TAHAP 5 — PELABELAN (LABELING / BINNING) — KHUSUS KLASIFIKASI
# Tujuan: membentuk target kelas price_level berdasarkan distribusi harga
# Catatan: dilakukan PER KOMODITAS agar adil antar skala harga

# Tahap 5.1 - Fungsi pelabelan berbasis kuartil
def label_price_quantile(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    return pd.cut(
        series,
        bins=[-np.inf, q1, q3, np.inf],
        labels=["Rendah", "Sedang", "Tinggi"]
    )

# Tahap 5.2 - Terapkan labeling per komoditas
df_long["price_level"] = (
    df_long
    .groupby("komoditas")["harga"]
    .transform(label_price_quantile)
)

# Tahap 5.3 - Validasi hasil labeling
print("Distribusi label price_level:")
print(df_long["price_level"].value_counts())

print("\nContoh data setelah pelabelan:")
df_long[["tanggal", "komoditas", "provinsi", "harga", "price_level"]].head()

# %%
# TAHAP 6 — TEMPORAL ORDERING (TIME-SERIES INTEGRITY)
# Tujuan: memastikan data harga tersusun kronologis dan valid sebagai time-series

# Tahap 6.1 - Pengurutan data time-series
df_long = df_long.sort_values(
    by=["komoditas", "provinsi", "tanggal"]
).reset_index(drop=True)

# Tahap 6.2 - Validasi urutan waktu (monotonic increasing)
validasi_waktu = (
    df_long
    .groupby(["komoditas", "provinsi"])["tanggal"]
    .apply(lambda x: x.is_monotonic_increasing)
)

print("Seluruh time-series terurut dengan benar:", validasi_waktu.all())

# Tahap 6.3 - Validasi duplikasi timestamp
duplikasi_waktu = df_long.duplicated(
    subset=["komoditas", "provinsi", "tanggal"]
).sum()

print("Jumlah duplikasi waktu:", duplikasi_waktu)

# Tahap 6.4 - Diagnostik frekuensi waktu (selisih tanggal)
contoh_frekuensi = (
    df_long
    .sort_values("tanggal")
    .groupby(["komoditas", "provinsi"])["tanggal"]
    .apply(lambda x: x.diff().dt.days.value_counts().head(3))
    .reset_index()
    .rename(columns={
        "level_2": "selisih_hari",
        "tanggal": "frekuensi"
    })
)

print("\nContoh selisih tanggal (hari):")
print(contoh_frekuensi.head(10))

# Tahap 6.5 - Statistik panjang setiap time-series
panjang_series = (
    df_long
    .groupby(["komoditas", "provinsi"])
    .size()
    .describe()
)

print("\nStatistik panjang time-series:")
print(panjang_series)

# %%
# %%
# TAHAP 7 — FEATURE ENGINEERING
# Tujuan: membentuk fitur prediktif dari data time-series harga

# TAHAP 7.1 — DATE-BASED FEATURES
df_long["year"] = df_long["tanggal"].dt.year
df_long["month"] = df_long["tanggal"].dt.month
df_long["week"] = df_long["tanggal"].dt.isocalendar().week.astype(int)
df_long["day"] = df_long["tanggal"].dt.day
df_long["day_of_week"] = df_long["tanggal"].dt.dayofweek  # 0 = Senin


print("Contoh fitur berbasis tanggal:")
print(
    df_long[["tanggal", "year", "month", "week", "day", "day_of_week"]].head()
)


# TAHAP 7.2 — LAG FEATURES
LAG_WINDOWS = [1, 7, 30]

for lag in LAG_WINDOWS:
    df_long[f"harga_lag_{lag}"] = (
        df_long
        .groupby(["komoditas", "provinsi"])["harga"]
        .shift(lag)
    )

print("\nContoh fitur lag:")
print(
    df_long[
        ["harga", "harga_lag_1", "harga_lag_7", "harga_lag_30"]
    ].head(10)
)


# TAHAP 7.3 — ROLLING STATISTICS
ROLLING_WINDOWS = [7, 30]

for window in ROLLING_WINDOWS:
    df_long[f"harga_roll_mean_{window}"] = (
        df_long
        .groupby(["komoditas", "provinsi"])["harga"]
        .transform(lambda x: x.rolling(window=window).mean())
    )

print("\nContoh fitur rolling mean:")
df_long[
    ["harga", "harga_roll_mean_7", "harga_roll_mean_30"]
].head(15)

# %%
# TAHAP 8 — OUTLIER DETECTION & HANDLING
# Tujuan: mendeteksi dan menangani nilai ekstrem pada data harga
from scipy.stats import zscore

# TAHAP 8.1 — DETEKSI OUTLIER (IQR)
Q1 = df_long["harga"].quantile(0.25)
Q3 = df_long["harga"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Batas bawah IQR:", round(lower_bound, 2))
print("Batas atas IQR :", round(upper_bound, 2))

# Simpan statistik sebelum clipping
harga_sebelum = df_long["harga"].describe()

# ✅ CLIPPING (TIDAK ADA BARIS DIHAPUS)
df_long["harga"] = df_long["harga"].clip(
    lower=lower_bound,
    upper=upper_bound)

# TAHAP 8.5 — VALIDASI SETELAH PENANGANAN
harga_sesudah = df_long["harga"].describe()

print("\nStatistik harga SEBELUM clipping:")
print(harga_sebelum.round(2))

print("\nStatistik harga SETELAH clipping:")
print(harga_sesudah.round(2))

print("\nJumlah baris data (tetap):", df_long.shape[0])

# %%
# TAHAP 9 — ENCODING VARIABEL KATEGORIKAL
# Tujuan: membuat data kompatibel dengan berbagai jenis model machine learning

from sklearn.preprocessing import LabelEncoder

# TAHAP 9.1 — Identifikasi variabel kategorikal
kolom_kategorikal = ["komoditas", "provinsi"]

print("Kolom kategorikal:", kolom_kategorikal)

# OPSI A — ONE-HOT ENCODING
# Digunakan untuk Linear Model / Neural Network
df_onehot = pd.get_dummies(
    df_long,
    columns=kolom_kategorikal,
    drop_first=True
)

print("\nUkuran dataset setelah One-Hot Encoding:")
print(df_onehot.shape)

# OPSI B — LABEL ENCODING
# Digunakan untuk Tree-based Model
df_label = df_long.copy()

encoder_dict = {}

for col in kolom_kategorikal:
    le = LabelEncoder()
    df_label[col] = le.fit_transform(df_label[col])
    encoder_dict[col] = le

    print(f"Jumlah kategori {col}:", len(le.classes_))


print("\nContoh hasil Label Encoding:")
df_label[kolom_kategorikal].head()

# %%
# MASTER DATA
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data_Pertanian"

# %%
# LOAD DATA HARGA BAHAN PANGAN (MASTER)

HARGA_DIR = DATA_DIR / "Harga Bahan Pangan" / "train"

df_list = []

for file in HARGA_DIR.glob("*.csv"):
    # Nama file = komoditas
    komoditas = file.stem.replace("_", " ").strip().lower()

    df_tmp = pd.read_csv(file)

    # =============================
    # DETEKSI KOLOM TANGGAL
    # =============================
    kandidat_tanggal = [
        c for c in df_tmp.columns
        if c.lower() in ["tanggal", "date", "tgl", "waktu"]
    ]

    if not kandidat_tanggal:
        raise KeyError(f"Tidak ditemukan kolom tanggal di {file.name}")

    kolom_tanggal = kandidat_tanggal[0]
    df_tmp = df_tmp.rename(columns={kolom_tanggal: "tanggal"})
    df_tmp["tanggal"] = pd.to_datetime(df_tmp["tanggal"], errors="coerce")

    # =============================
    # MELT (WIDE → LONG)
    # =============================
    kolom_provinsi = [
        c for c in df_tmp.columns if c != "tanggal"
    ]

    df_long_tmp = df_tmp.melt(
        id_vars="tanggal",
        value_vars=kolom_provinsi,
        var_name="provinsi",
        value_name="harga"
    )

    df_long_tmp["komoditas"] = komoditas

    # Normalisasi teks
    df_long_tmp["provinsi"] = df_long_tmp["provinsi"].str.strip().str.lower()
    df_long_tmp["komoditas"] = df_long_tmp["komoditas"].str.strip().str.lower()

    df_list.append(df_long_tmp)

# =============================
# GABUNGKAN SEMUA KOMODITAS
# =============================
df_master = pd.concat(df_list, ignore_index=True)

print("Data harga pangan berhasil dimuat")
print("Ukuran df_master:", df_master.shape)
print("Kolom:", df_master.columns.tolist())

# %%
# LOAD DATA MATA UANG

KURS_DIR = DATA_DIR / "Mata Uang"

kurs_files = {
    "MYRUSD": "MYRUSD=X.csv",
    "SGDUSD": "SGDUSD=X.csv",
    "THBUSD": "THBUSD=X.csv",
    "USDIDR": "USDIDR=X.csv"
}

df_kurs_list = []

for kode, fname in kurs_files.items():
    df_tmp = pd.read_csv(KURS_DIR / fname)
    df_tmp["Date"] = pd.to_datetime(df_tmp["Date"])

    df_tmp = df_tmp[["Date", "Close"]].rename(columns={
        "Date": "tanggal",
        "Close": f"kurs_{kode.lower()}"
    })

    df_kurs_list.append(df_tmp)

df_kurs = df_kurs_list[0]
for df_tmp in df_kurs_list[1:]:
    df_kurs = df_kurs.merge(df_tmp, on="tanggal", how="outer")

print("Data kurs dimuat:", df_kurs.shape)

# %%
# LOAD GLOBAL COMMODITY PRICE

GLOBAL_DIR = DATA_DIR / "Global Commodity Price"

commodity_files = {
    "crude_oil": "Crude Oil WTI Futures Historical Data.csv",
    "natural_gas": "Natural Gas Futures Historical Data.csv",
    "coal": "Newcastle Coal Futures Historical Data.csv",
    "palm_oil": "Palm Oil Futures Historical Data.csv",
    "sugar": "US Sugar 11 Futures Historical Data.csv",
    "wheat": "US Wheat Futures Historical Data.csv"
}

df_global_list = []

for name, fname in commodity_files.items():
    df_tmp = pd.read_csv(GLOBAL_DIR / fname)
    df_tmp["Date"] = pd.to_datetime(df_tmp["Date"])

    df_tmp = df_tmp[["Date", "Price"]].rename(columns={
        "Date": "tanggal",
        "Price": f"global_{name}"
    })

    df_global_list.append(df_tmp)

df_global = df_global_list[0]
for df_tmp in df_global_list[1:]:
    df_global = df_global.merge(df_tmp, on="tanggal", how="outer")

print("Data global commodity dimuat:", df_global.shape)

# %%
# LOAD GOOGLE TREND

TREND_DIR = DATA_DIR / "Google Trend"
df_trend_list = []

for folder in TREND_DIR.iterdir():
    if not folder.is_dir():
        continue

    komoditas = folder.name.strip().lower()

    for file in folder.glob("*.csv"):
        provinsi = file.stem.strip().lower()

        df_tmp = pd.read_csv(file)
        if df_tmp.shape[1] < 2:
            continue

        df_tmp = df_tmp.rename(columns={
            df_tmp.columns[0]: "tanggal",
            df_tmp.columns[1]: "google_trend"
        })

        df_tmp["tanggal"] = pd.to_datetime(df_tmp["tanggal"], errors="coerce")
        df_tmp["komoditas"] = komoditas
        df_tmp["provinsi"] = provinsi

        df_trend_list.append(df_tmp)

if not df_trend_list:
    raise ValueError("Data Google Trend kosong")

df_trend = pd.concat(df_trend_list, ignore_index=True)

print("Google Trend dimuat:", df_trend.shape)

# %%
# MERGE DATA EKSTERNAL

df_master = (
    df_master
    .merge(df_kurs, on="tanggal", how="left")
    .merge(df_global, on="tanggal", how="left")
    .merge(df_trend, on=["tanggal", "komoditas", "provinsi"], how="left")
)

print("Final master shape:", df_master.shape)
print("Kolom setelah merge:", df_master.columns.tolist())

# %%
# FORWARD FILL GOOGLE TREND (tanpa menghapus kolom groupby)
df_master = df_master.sort_values("tanggal").reset_index(drop=True)

# Forward fill per grup komoditas & provinsi
for col in ['google_trend']:
    if col in df_master.columns:
        df_master[col] = (
            df_master
            .groupby(["komoditas", "provinsi"], group_keys=False)[col]
            .ffill()
        )
# %%
# NORMALIZE KOMODITAS (lowercase) sebelum labeling
if "komoditas" in df_master.columns:
    df_master["komoditas"] = df_master["komoditas"].str.strip().str.lower()
else:
    print("ERROR: Kolom 'komoditas' tidak ditemukan!")
    print("Kolom yang ada:", df_master.columns.tolist())
    raise KeyError("Kolom komoditas hilang")

# TAMBAH PRICE_LEVEL KE df_master (PENTING untuk Tahap 10)
# Gunakan fungsi labeling yang sama dari Tahap 5
def label_price_quantile(series):      
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    return pd.cut(
        series,
        bins=[-np.inf, q1, q3, np.inf],
        labels=["Rendah", "Sedang", "Tinggi"]
    )

df_master["price_level"] = (           
    df_master
    .groupby("komoditas")["harga"]
    .transform(label_price_quantile)
)

# %%
OUTPUT_DIR = BASE_DIR / "Dataset_Final"
OUTPUT_DIR.mkdir(exist_ok=True)

output_file = OUTPUT_DIR / "master_dataset.csv"
df_master.to_csv(output_file, index=False)

print("MASTER DATASET BERHASIL DISIMPAN")
print(output_file)


# %%
# ========================================================
# PERBAIKAN TAHAP 10 - 12 (MENGGUNAKAN df_master)
# ========================================================
# Memastikan data eksternal (Kurs, Global, Trends) masuk ke model

from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# 1. Gunakan df_master yang sudah lengkap, tapi lakukan encoding dulu
df_final = df_master.copy()

le_komo = LabelEncoder()
le_prov = LabelEncoder()

df_final['komoditas'] = le_komo.fit_transform(df_final['komoditas'])
df_final['provinsi'] = le_prov.fit_transform(df_final['provinsi'])

# 2. Definisikan Fitur
# Masukkan semua kolom kurs, global, dan trend yang sudah di-merge
kolom_eksternal = [c for c in df_final.columns 
                   if c.startswith(('kurs_', 'global_', 'google_trend'))]

fitur_waktu = ['month', 'week', 'day', 'day_of_week']
fitur_lag = [c for c in df_final.columns if c.startswith('harga_lag_')]

# Jika kolom waktu belum ada, buat terlebih dahulu
if 'month' not in df_final.columns:
    df_final["month"] = df_final["tanggal"].dt.month
    df_final["week"] = df_final["tanggal"].dt.isocalendar().week.astype(int)
    df_final["day"] = df_final["tanggal"].dt.day
    df_final["day_of_week"] = df_final["tanggal"].dt.dayofweek

# Jika fitur lag belum ada, buat terlebih dahulu
if not fitur_lag:
    LAG_WINDOWS = [1, 7, 30]
    for lag in LAG_WINDOWS:
        df_final[f"harga_lag_{lag}"] = (
            df_final
            .groupby(["komoditas", "provinsi"])["harga"]
            .shift(lag)
        )
    fitur_lag = [c for c in df_final.columns if c.startswith('harga_lag_')]

feature_list = ['komoditas', 'provinsi'] + fitur_waktu + fitur_lag + kolom_eksternal

print("Fitur yang digunakan:")
print(feature_list)

# 3. Handling NaN hasil merge (Sangat Penting!)
# Data eksternal sering kosong di hari libur, kita isi dengan nilai sebelumnya
df_final = df_final.sort_values('tanggal')
df_final[kolom_eksternal] = (
    df_final
    .groupby(['komoditas', 'provinsi'])[kolom_eksternal]
    .ffill()
    .bfill()
)

df_final = df_final.dropna(subset=feature_list + ['price_level'])

print("Ukuran data setelah handling NaN:", df_final.shape)

# 4. Split Data (TIME-BASED)
train_mask = df_final["tanggal"].dt.year < 2023
df_train = df_final.loc[train_mask].copy()
df_test = df_final.loc[~train_mask].copy()

print("Ukuran data train:", df_train.shape)
print("Ukuran data test :", df_test.shape)

# 5. Scaling (Hanya untuk fitur kontinu)
scaler = StandardScaler()
fitur_kontinu = fitur_lag + kolom_eksternal

print("\nDEBUG: Sebelum konversi ke numeric")
print("fitur_kontinu:", fitur_kontinu)
print("Kolom yang ada di df_train:", df_train.columns.tolist())

# CEK dulu - hanya konversi jika ada NaN atau string
for col in fitur_kontinu:
    if col in df_train.columns:
        # Cek type
        if df_train[col].dtype == 'object':
            print(f"  {col}: tipe object, sample: {df_train[col].iloc[0]}")
            df_train[col] = pd.to_numeric(df_train[col].astype(str).str.replace('.', '', 1).str.replace(',', '.'), errors='coerce')
            df_test[col] = pd.to_numeric(df_test[col].astype(str).str.replace('.', '', 1).str.replace(',', '.'), errors='coerce')
        else:
            print(f"  {col}: tipe {df_train[col].dtype}")

print("\nDEBUG: Jumlah NaN per kolom setelah konversi")
nan_counts = df_train[fitur_kontinu].isna().sum()
print(nan_counts)

# FILL NaN dengan median (lebih aman dari drop semua)
print("\nMengisi NaN dengan median...")
for col in fitur_kontinu:
    if col in df_train.columns and df_train[col].isna().sum() > 0:
        median_val = df_train[col].median()
        print(f"  {col}: fill {df_train[col].isna().sum()} NaN dengan median {median_val:.2f}")
        df_train[col].fillna(median_val, inplace=True)
        df_test[col].fillna(median_val, inplace=True)

print("\nUkuran setelah fill NaN:")
print("Ukuran data train:", df_train.shape)
print("Ukuran data test :", df_test.shape)

# FIT hanya pada TRAIN
scaler.fit(df_train[fitur_kontinu])

df_train[fitur_kontinu] = scaler.transform(df_train[fitur_kontinu])
df_test[fitur_kontinu] = scaler.transform(df_test[fitur_kontinu])

print("\nStatistik fitur kontinu setelah scaling (TRAIN):")
print(df_train[fitur_kontinu].describe().round(2))

# 6. Encode Target
le_target = LabelEncoder()
y_train = le_target.fit_transform(df_train['price_level'])
y_test = le_target.transform(df_test['price_level'])

X_train = df_train[feature_list].copy()
X_test = df_test[feature_list].copy()

print("\nMapping kelas target:")
for i, label in enumerate(le_target.classes_):
    print(f"{label} -> {i}")

print("\nUkuran dataset final:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)

# 7. Simpan Artefak Lengkap
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

joblib.dump(scaler, ARTIFACT_DIR / "scaler.pkl")
joblib.dump(le_target, ARTIFACT_DIR / "target_encoder.pkl")
joblib.dump(le_komo, ARTIFACT_DIR / "komoditas_encoder.pkl")
joblib.dump(le_prov, ARTIFACT_DIR / "provinsi_encoder.pkl")
joblib.dump(feature_list, ARTIFACT_DIR / "feature_list.pkl")

print("\nSemua artefak berhasil disimpan ke:", ARTIFACT_DIR)


# %%
# TAHAP M1 — LOAD DATA EKSTERNAL
# Tujuan: memuat data Kurs, Global Price, dan Google Trends

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_EKSTERNAL_DIR = BASE_DIR / "Data_Eksternal"

# Load Kurs
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data_Pertanian"
KURS_DIR = DATA_DIR / "Mata Uang"

kurs_files = {
    "myrusd": "MYRUSD=X.csv",
    "sgdusd": "SGDUSD=X.csv",
    "thbusd": "THBUSD=X.csv",
    "usdidr": "USDIDR=X.csv"
}

df_kurs_list = []

for kode, fname in kurs_files.items():
    file_path = KURS_DIR / fname
    if not file_path.exists():
        raise FileNotFoundError(f"File kurs tidak ditemukan: {file_path}")

    df_tmp = pd.read_csv(file_path)
    df_tmp["Date"] = pd.to_datetime(df_tmp["Date"], errors="coerce")

    df_tmp = df_tmp[["Date", "Close"]].rename(columns={
        "Date": "tanggal",
        "Close": f"kurs_{kode}"
    })

    df_kurs_list.append(df_tmp)

df_kurs = df_kurs_list[0]
for df_tmp in df_kurs_list[1:]:
    df_kurs = df_kurs.merge(df_tmp, on="tanggal", how="outer")

print("Data kurs dimuat:", df_kurs.shape)

# Load Global Price
GLOBAL_DIR = DATA_DIR / "Global Commodity Price"

commodity_files = {
    "crude_oil": "Crude Oil WTI Futures Historical Data.csv",
    "natural_gas": "Natural Gas Futures Historical Data.csv",
    "coal": "Newcastle Coal Futures Historical Data.csv",
    "palm_oil": "Palm Oil Futures Historical Data.csv",
    "sugar": "US Sugar 11 Futures Historical Data.csv",
    "wheat": "US Wheat Futures Historical Data.csv"
}

df_global_list = []

for name, fname in commodity_files.items():
    file_path = GLOBAL_DIR / fname
    if not file_path.exists():
        raise FileNotFoundError(f"File global commodity tidak ditemukan: {file_path}")

    df_tmp = pd.read_csv(file_path)
    df_tmp["Date"] = pd.to_datetime(df_tmp["Date"], errors="coerce")

    df_tmp = df_tmp[["Date", "Price"]].rename(columns={
        "Date": "tanggal",
        "Price": f"global_{name}"
    })

    df_global_list.append(df_tmp)

df_global = df_global_list[0]
for df_tmp in df_global_list[1:]:
    df_global = df_global.merge(df_tmp, on="tanggal", how="outer")

print("Data global commodity dimuat:", df_global.shape)

# Load Google Trends
TREND_DIR = DATA_DIR / "Google Trend"

df_trend_list = []

for folder_komoditas in TREND_DIR.iterdir():
    if not folder_komoditas.is_dir():
        continue

    komoditas = folder_komoditas.name.strip().lower()

    for file_csv in folder_komoditas.glob("*.csv"):
        provinsi = file_csv.stem.strip().lower()

        df_tmp = pd.read_csv(file_csv)

        # Validasi minimal kolom
        if df_tmp.shape[1] < 2:
            continue

        df_tmp = df_tmp.rename(columns={
            df_tmp.columns[0]: "tanggal",
            df_tmp.columns[1]: "google_trend"
        })

        df_tmp["tanggal"] = pd.to_datetime(df_tmp["tanggal"], errors="coerce")
        df_tmp["komoditas"] = komoditas
        df_tmp["provinsi"] = provinsi

        df_trend_list.append(
            df_tmp[["tanggal", "komoditas", "provinsi", "google_trend"]]
        )

if len(df_trend_list) == 0:
    raise ValueError("❌ Tidak ada data Google Trend yang berhasil dimuat")

df_trends = pd.concat(df_trend_list, ignore_index=True)

print("Data Google Trends dimuat:", df_trends.shape)


# %%
#===============================================================================
#MODELING 
#===============================================================================

# %% TAHAP MOD-1: TRAINING 3 ALGORITMA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Definisi model
models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000, multi_class='multinomial'),
    "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

# Penampung hasil
results = {}

for name, model in models.items():
    print(f"\n--- Melatih {name} ---")
    model.fit(X_train, y_train)
    
    # Prediksi data test (Tahun 2023)
    y_pred = model.predict(X_test)
    
    # Evaluasi
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    
    print(f"Akurasi {name}: {acc:.4f}")
    # Gunakan le_target.classes_ agar muncul label 'Rendah', 'Sedang', 'Tinggi'
    print(classification_report(y_test, y_pred, target_names=le_target.classes_))
    
    # %% TAHAP MOD-2: PERBANDINGAN
print("\n=== RINGKASAN AKURASI ===")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")

# Cari yang terbaik secara otomatis
best_model_name = max(results, key=results.get)
print(f"\nModel Terbaik: {best_model_name}")
# %%

# %% TAHAP MOD-3: PENYIMPANAN
# Simpan model terbaik
joblib.dump(models[best_model_name], ARTIFACT_DIR / "best_model.pkl")

print(f"✅ Model {best_model_name} berhasil disimpan di folder artifacts.")