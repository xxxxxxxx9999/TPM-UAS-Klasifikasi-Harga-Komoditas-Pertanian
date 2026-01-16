import os
from typing import Dict, List, Optional, Tuple

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------
# Page config + lightweight styling
# ----------------------------
st.set_page_config(
    page_title="Prediksi Level Harga Pangan",
    page_icon="🌾",
    layout="wide",
)

st.markdown(
    """
<style>
  .small-muted { color: rgba(255,255,255,0.65); font-size: 0.9rem; }
  .kpi-card { padding: 14px 16px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.03); }
  .kpi-title { font-size: 0.85rem; color: rgba(255,255,255,0.70); margin-bottom: 6px; }
  .kpi-value { font-size: 1.6rem; font-weight: 700; line-height: 1.1; }
  .kpi-sub { font-size: 0.85rem; color: rgba(255,255,255,0.65); margin-top: 6px; }
  .pill { display: inline-block; padding: 4px 10px; border-radius: 999px; border: 1px solid rgba(255,255,255,0.16); background: rgba(255,255,255,0.06); font-size: 0.85rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------
# Paths (sesuaikan bila kamu taruh di folder lain)
# ----------------------------
ART_DIR = "artifacts"
DEFAULT_FEATURE_STORE = os.path.join("Dataset_Final", "master_dataset.csv")
# Jika kamu punya feature store siap pakai (disarankan), taruh misal:
# data/feature_store.csv.gz atau Dataset_Final/feature_store.csv.gz
CANDIDATE_STORES = [
    os.path.join("data", "feature_store.csv.gz"),
    os.path.join("Dataset_Final", "feature_store.csv.gz"),
    os.path.join("data", "feature_store.csv"),
    os.path.join("Dataset_Final", "feature_store.csv"),
    DEFAULT_FEATURE_STORE,
]


# ----------------------------
# Utilities
# ----------------------------

def _safe_read_csv(path: str) -> pd.DataFrame:
    # gzip csv supported by pandas read_csv
    df = pd.read_csv(path)
    # normalize common date columns
    for c in ["tanggal", "Date", "date", "DATE"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            if c != "tanggal":
                df.rename(columns={c: "tanggal"}, inplace=True)
            break
    return df


def _find_first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def _fmt_float(x: Optional[float], nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    try:
        return f"{float(x):,.{nd}f}".replace(",", ".")
    except Exception:
        return str(x)


def _kpi(title: str, value: str, sub: str = ""):
    st.markdown(
        f"""
<div class="kpi-card">
  <div class="kpi-title">{title}</div>
  <div class="kpi-value">{value}</div>
  <div class="kpi-sub">{sub}</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ----------------------------
# Load artifacts (model + preprocess)
# ----------------------------

@st.cache_resource
def load_artifacts() -> Dict[str, object]:
    required = [
        "best_model.pkl",
        "scaler.pkl",
        "target_encoder.pkl",
        "komoditas_encoder.pkl",
        "provinsi_encoder.pkl",
        "feature_list.pkl",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(ART_DIR, f))]
    if missing:
        raise FileNotFoundError(
            "Artifacts tidak lengkap. Pastikan file berikut ada di folder 'artifacts/': "
            + ", ".join(missing)
        )

    model = joblib.load(os.path.join(ART_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(ART_DIR, "scaler.pkl"))
    y_enc = joblib.load(os.path.join(ART_DIR, "target_encoder.pkl"))
    kom_enc = joblib.load(os.path.join(ART_DIR, "komoditas_encoder.pkl"))
    prov_enc = joblib.load(os.path.join(ART_DIR, "provinsi_encoder.pkl"))
    feat_list = joblib.load(os.path.join(ART_DIR, "feature_list.pkl"))

    if not isinstance(feat_list, (list, tuple)):
        raise ValueError("feature_list.pkl harus berisi list nama kolom fitur.")

    return {
        "model": model,
        "scaler": scaler,
        "y_enc": y_enc,
        "kom_enc": kom_enc,
        "prov_enc": prov_enc,
        "feat_list": list(feat_list),
    }


# ----------------------------
# Feature store / dataset loader
# ----------------------------

@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    df = _safe_read_csv(path)

    # Standardize potential column names (your pipeline often uses these)
    rename_map = {
        "Komoditas": "komoditas",
        "Provinsi": "provinsi",
        "harga": "harga",
        "Harga": "harga",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)

    # ensure datetime
    if "tanggal" in df.columns:
        df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")

    # normalize strings
    for c in ["komoditas", "provinsi"]:
        if c in df.columns and not _is_numeric_series(df[c]):
            df[c] = df[c].astype(str).str.strip().str.lower()

    return df


def infer_continuous_cols(feat_list: List[str]) -> List[str]:
    """Heuristik kolom kontinu seperti di training kamu: lag + eksternal."""
    cont = []
    for c in feat_list:
        cl = c.lower()
        if (
            cl.startswith("lag_")
            or "rolling" in cl
            or cl.startswith("kurs_")
            or cl.startswith("global_")
            or "google_trend" in cl
            or cl.startswith("trend_")
        ):
            cont.append(c)
    return cont


def build_X(df_rows: pd.DataFrame, feat_list: List[str], scaler) -> np.ndarray:
    """Ambil fitur sesuai feature_list + scaling sesuai scaler."""
    missing = [c for c in feat_list if c not in df_rows.columns]
    if missing:
        raise KeyError(
            "Dataset belum punya kolom fitur lengkap. Kolom hilang: " + ", ".join(missing[:20])
            + (" ..." if len(missing) > 20 else "")
        )

    X_df = df_rows[feat_list].copy()

    # Convert to numeric where possible
    for c in X_df.columns:
        if not _is_numeric_series(X_df[c]):
            X_df[c] = pd.to_numeric(X_df[c], errors="coerce")

    # Scaling strategy:
    # - Jika scaler dilatih pada semua fitur: transform seluruh matrix
    # - Jika scaler dilatih hanya pada subset kontinu: transform subset kontinu saja
    n_expected = getattr(scaler, "n_features_in_", None)
    cont_cols = infer_continuous_cols(feat_list)

    if n_expected is None:
        # fallback: attempt full transform
        return scaler.transform(X_df)

    if n_expected == len(feat_list):
        return scaler.transform(X_df)

    if n_expected == len(cont_cols) and len(cont_cols) > 0:
        X_scaled = X_df.to_numpy(dtype=float, copy=True)
        # build index map
        idx = [feat_list.index(c) for c in cont_cols]
        cont_mat = X_df[cont_cols].to_numpy(dtype=float, copy=True)
        cont_scaled = scaler.transform(cont_mat)
        X_scaled[:, idx] = cont_scaled
        return X_scaled

    raise ValueError(
        f"Scaler expects {n_expected} features, tapi app tidak bisa mencocokkan. "
        f"feat_list={len(feat_list)}; cont_cols_guess={len(cont_cols)}. "
        "Solusi: simpan juga daftar kolom yang di-scale saat training, atau scale semua fitur saat training."
    )


def predict_with_proba(model, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    pred = model.predict(X)
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
        except Exception:
            proba = None
    return pred, proba


def decode_category_column(s: pd.Series, encoder) -> pd.Series:
    """Jika kolom berupa integer hasil encoding, decode ke label string."""
    if _is_numeric_series(s):
        vals = s.astype(int).to_numpy()
        try:
            decoded = encoder.inverse_transform(vals)
            return pd.Series(decoded).astype(str)
        except Exception:
            return s.astype(str)
    return s.astype(str)


def encode_label_to_id(label: str, encoder) -> int:
    """Encode label string ke integer sesuai LabelEncoder (komoditas/provinsi)."""
    lab = str(label).strip().lower()
    try:
        return int(encoder.transform([lab])[0])
    except Exception as e:
        # tampilkan contoh label yang valid
        classes = getattr(encoder, "classes_", [])
        sample = ", ".join([str(x) for x in list(classes)[:10]])
        raise ValueError(
            f"Label '{lab}' tidak dikenali encoder. Contoh label valid: {sample}{' ...' if len(classes) > 10 else ''}. Error: {e}"
        )
    return s.astype(str)


# ----------------------------
# App
# ----------------------------

def main():
    st.title("🌾 Prediksi Level Harga Bahan Pangan")
    st.write(
        "Dashboard deployment (Streamlit) untuk memprediksi **Rendah / Sedang / Tinggi** berdasarkan fitur model."
    )

    # Load artifacts
    try:
        art = load_artifacts()
    except Exception as e:
        st.error(f"Gagal load artifacts: {e}")
        st.stop()

    model = art["model"]
    scaler = art["scaler"]
    y_enc = art["y_enc"]
    kom_enc = art["kom_enc"]
    prov_enc = art["prov_enc"]
    feat_list = art["feat_list"]

    # Choose dataset source
    st.sidebar.header("📦 Sumber Data")
    existing = _find_first_existing(CANDIDATE_STORES)
    use_existing = st.sidebar.toggle(
        "Pakai file dataset lokal (jika ada)",
        value=True if existing else False,
        help="Jika repo kamu sudah menyertakan feature_store.csv(.gz) atau master_dataset.csv, aktifkan ini.",
    )

    uploaded = None
    if not use_existing:
        uploaded = st.sidebar.file_uploader(
            "Upload feature store / dataset (CSV)",
            type=["csv", "gz"],
            help="Upload CSV yang punya kolom: tanggal, komoditas, provinsi, serta fitur-fitur sesuai feature_list.pkl.",
        )

    df = None
    data_path = None
    try:
        if use_existing:
            if not existing:
                st.warning(
                    "Tidak menemukan dataset lokal. Upload file CSV lewat sidebar atau tambahkan file feature_store.csv(.gz) ke repo."
                )
            else:
                data_path = existing
                df = load_dataset(existing)
        else:
            if uploaded is not None:
                df = pd.read_csv(uploaded)
                for c in ["tanggal", "Date", "date"]:
                    if c in df.columns:
                        df[c] = pd.to_datetime(df[c], errors="coerce")
                        if c != "tanggal":
                            df.rename(columns={c: "tanggal"}, inplace=True)
                        break
                for c in ["komoditas", "provinsi"]:
                    if c in df.columns and not _is_numeric_series(df[c]):
                        df[c] = df[c].astype(str).str.strip().str.lower()
    except Exception as e:
        st.error(f"Gagal membaca dataset: {e}")
        st.stop()

    if df is None:
        st.info(
            "Tambahkan salah satu file berikut ke repo (disarankan): `data/feature_store.csv.gz` atau `Dataset_Final/feature_store.csv.gz`.\n\n"
            "Atau upload CSV dari sidebar."
        )
        st.stop()

    # Basic validation
    if "tanggal" not in df.columns:
        st.error("Dataset wajib punya kolom `tanggal`.")
        st.stop()

    # Build display columns for selectors
    df_sel = df.copy()

    if "komoditas" in df_sel.columns:
        df_sel["komoditas_display"] = decode_category_column(df_sel["komoditas"], kom_enc).str.lower()
    else:
        df_sel["komoditas_display"] = "(unknown)"

    if "provinsi" in df_sel.columns:
        df_sel["provinsi_display"] = decode_category_column(df_sel["provinsi"], prov_enc).str.lower()
    else:
        df_sel["provinsi_display"] = "(unknown)"

    # Sidebar filters
    st.sidebar.header("🎯 Filter")
    kom_opts = sorted(df_sel["komoditas_display"].dropna().unique().tolist())
    prov_opts = sorted(df_sel["provinsi_display"].dropna().unique().tolist())

    if not kom_opts or not prov_opts:
        st.error("Kolom komoditas/provinsi tidak tersedia (atau tidak bisa didecode).")
        st.stop()

    kom_choice = st.sidebar.selectbox("Komoditas", kom_opts)
    prov_choice = st.sidebar.selectbox("Provinsi", prov_opts)

    # Date range for that selection
    sub = df_sel[(df_sel["komoditas_display"] == kom_choice) & (df_sel["provinsi_display"] == prov_choice)].copy()
    sub = sub.dropna(subset=["tanggal"]).sort_values("tanggal")

    if sub.empty:
        st.warning("Tidak ada baris data untuk pilihan itu.")
        st.stop()

    min_d = sub["tanggal"].min().date()
    max_d = sub["tanggal"].max().date()

    mode = st.sidebar.radio(
        "Mode",
        ["Prediksi 1 Tanggal", "Analisis Rentang", "Input Manual", "Batch (Upload fitur)"],
        index=0,
    )

    # Header / metadata
    st.caption(
        f"Dataset: **{data_path or 'upload'}** · Rentang tanggal tersedia untuk pilihan ini: **{min_d}** s/d **{max_d}**"
    )

    # ----------------------------
    # Mode 1: Single date
    # ----------------------------
    if mode == "Prediksi 1 Tanggal":
        chosen_date = st.sidebar.date_input("Tanggal", value=max_d, min_value=min_d, max_value=max_d)

        row = sub[sub["tanggal"].dt.date == chosen_date]
        if row.empty:
            st.error("Tidak ada data untuk tanggal itu (untuk komoditas & provinsi yang dipilih).")
            st.stop()

        # If duplicate dates, keep last
        row = row.sort_values("tanggal").tail(1)

        # Prepare X
        try:
            X = build_X(row, feat_list, scaler)
        except Exception as e:
            st.error(f"Gagal membangun fitur untuk prediksi: {e}")
            with st.expander("Debug: contoh kolom dataset"):
                st.write(list(df.columns))
            st.stop()

        pred, proba = predict_with_proba(model, X)
        pred_id = int(pred[0])

        try:
            pred_label = y_enc.inverse_transform([pred_id])[0]
        except Exception:
            pred_label = str(pred_id)

        confidence = None
        proba_df = None
        if proba is not None:
            probs = proba[0]
            confidence = float(np.max(probs))
            try:
                labels = list(y_enc.inverse_transform(np.arange(len(probs))))
            except Exception:
                labels = [str(i) for i in range(len(probs))]
            proba_df = (
                pd.DataFrame({"Label": labels, "Probabilitas": probs})
                .sort_values("Probabilitas", ascending=False)
                .reset_index(drop=True)
            )

        # KPIs
        harga_today = row["harga"].iloc[0] if "harga" in row.columns else None

        # delta vs previous day (if available)
        prev = sub[sub["tanggal"] < pd.Timestamp(chosen_date)].sort_values("tanggal").tail(1)
        delta_txt = ""
        if harga_today is not None and not prev.empty and "harga" in prev.columns:
            prev_price = prev["harga"].iloc[0]
            if pd.notna(prev_price) and pd.notna(harga_today):
                delta = float(harga_today) - float(prev_price)
                pct = (delta / float(prev_price)) * 100 if float(prev_price) != 0 else np.nan
                sign = "+" if delta >= 0 else ""
                delta_txt = f"Δ vs hari sebelumnya: {sign}{_fmt_float(delta, 2)} ({sign}{_fmt_float(pct, 2)}%)"

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            _kpi("Harga", _fmt_float(harga_today, 0) if harga_today is not None else "-", delta_txt)
        with c2:
            _kpi("Prediksi Level", f"<span class='pill'>{pred_label}</span>", "")
        with c3:
            _kpi("Confidence", _fmt_float(confidence, 3) if confidence is not None else "-", "max probability")
        with c4:
            n_rows = sub.shape[0]
            _kpi("Data tersedia", f"{n_rows:,}", "baris untuk komoditas-provinsi ini")

        # Tabs
        tab1, tab2, tab3 = st.tabs(["📈 Tren Harga", "🎯 Probabilitas", "🧩 Fitur (ringkas)"])

        with tab1:
            plot_df = sub[["tanggal"] + (["harga"] if "harga" in sub.columns else [])].copy()
            plot_df["tanggal"] = pd.to_datetime(plot_df["tanggal"])
            plot_df = plot_df.dropna(subset=["tanggal"])

            if "harga" in plot_df.columns:
                base = alt.Chart(plot_df).mark_line().encode(
                    x=alt.X("tanggal:T", title="Tanggal"),
                    y=alt.Y("harga:Q", title="Harga"),
                    tooltip=[alt.Tooltip("tanggal:T"), alt.Tooltip("harga:Q")],
                )

                highlight = alt.Chart(pd.DataFrame({"tanggal": [pd.Timestamp(chosen_date)], "harga": [harga_today]})).mark_point(size=120).encode(
                    x="tanggal:T",
                    y="harga:Q",
                    tooltip=[alt.Tooltip("tanggal:T"), alt.Tooltip("harga:Q")],
                )

                st.altair_chart(base + highlight, use_container_width=True)
            else:
                st.info("Kolom `harga` tidak ada di dataset, jadi chart harga tidak ditampilkan.")

        with tab2:
            if proba_df is None:
                st.info("Model ini tidak menyediakan `predict_proba`. (Masih bisa prediksi label.)")
            else:
                chart = (
                    alt.Chart(proba_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("Probabilitas:Q", title="Probabilitas"),
                        y=alt.Y("Label:N", sort="-x", title="Kelas"),
                        tooltip=["Label", alt.Tooltip("Probabilitas:Q", format=".4f")],
                    )
                )
                st.altair_chart(chart, use_container_width=True)
                st.dataframe(proba_df, use_container_width=True)

        with tab3:
            # show key feature values (raw)
            feat_show = feat_list.copy()
            # keep compact
            priority = [c for c in feat_show if ("lag_" in c.lower() or "google_trend" in c.lower() or c.lower().startswith("kurs_") or c.lower().startswith("global_"))]
            time_feats = [c for c in feat_show if c.lower() in {"month", "week", "day", "day_of_week", "year"}]
            cat_feats = [c for c in feat_show if c.lower() in {"komoditas", "provinsi"}]

            picked = (cat_feats + time_feats + priority)[:30]
            view = row[picked].T.reset_index()
            view.columns = ["Fitur", "Nilai"]
            st.dataframe(view, use_container_width=True, height=520)

            st.caption("Menampilkan subset fitur yang paling informatif (kategori, waktu, lag & eksternal).")

        # Download single prediction
        out_row = row.copy()
        out_row["predicted_class_id"] = pred_id
        out_row["predicted_label"] = pred_label
        if proba_df is not None:
            out_row["confidence"] = confidence
        csv_bytes = out_row.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download hasil (CSV)", data=csv_bytes, file_name="prediction_single.csv", mime="text/csv")

    # ----------------------------
    # Mode 2: Range analysis
    # ----------------------------
    elif mode == "Analisis Rentang":
        d1, d2 = st.sidebar.date_input(
            "Rentang tanggal",
            value=(max(min_d, (max_d - pd.Timedelta(days=30)).date()), max_d),
            min_value=min_d,
            max_value=max_d,
        )
        if isinstance(d1, tuple) or isinstance(d2, tuple):
            # streamlit sometimes returns tuple; safeguard
            st.warning("Input rentang tanggal tidak valid.")
            st.stop()

        start = pd.Timestamp(d1)
        end = pd.Timestamp(d2) + pd.Timedelta(days=1)  # inclusive

        rng = sub[(sub["tanggal"] >= start) & (sub["tanggal"] < end)].copy()
        if rng.empty:
            st.warning("Tidak ada data pada rentang tersebut.")
            st.stop()

        # if many rows, optionally sample for speed
        max_rows = st.sidebar.slider("Maks baris untuk diproses", 200, 5000, 2000, step=100)
        if len(rng) > max_rows:
            rng = rng.sort_values("tanggal").tail(max_rows)
            st.info(f"Data dipotong ke {max_rows} baris terbaru untuk performa.")

        try:
            X = build_X(rng, feat_list, scaler)
        except Exception as e:
            st.error(f"Gagal membangun fitur: {e}")
            st.stop()

        pred, proba = predict_with_proba(model, X)
        pred = pred.astype(int)
        try:
            labels = y_enc.inverse_transform(pred)
        except Exception:
            labels = pred.astype(str)

        out = rng[["tanggal", "komoditas_display", "provinsi_display"] + (["harga"] if "harga" in rng.columns else [])].copy()
        out.rename(columns={"komoditas_display": "komoditas", "provinsi_display": "provinsi"}, inplace=True)
        out["predicted_class_id"] = pred
        out["predicted_label"] = labels

        if proba is not None:
            out["confidence"] = np.max(proba, axis=1)

        # KPIs
        c1, c2, c3 = st.columns(3)
        with c1:
            _kpi("Jumlah hari", f"{len(out):,}")
        with c2:
            avg_conf = float(out["confidence"].mean()) if "confidence" in out.columns else None
            _kpi("Rata-rata confidence", _fmt_float(avg_conf, 3) if avg_conf is not None else "-")
        with c3:
            if "price_level" in rng.columns:
                # compare if ground-truth exists
                gt = rng["price_level"].astype(str).to_numpy()
                acc = float(np.mean(gt == out["predicted_label"].astype(str).to_numpy()))
                _kpi("Akurasi vs label", _fmt_float(acc, 3), "(hanya jika kolom price_level ada)")
            else:
                _kpi("Akurasi vs label", "-", "kolom price_level tidak ada")

        # Charts
        left, right = st.columns([1.2, 1])
        with left:
            if "harga" in out.columns:
                chart_df = out.copy()
                line = alt.Chart(chart_df).mark_line().encode(
                    x=alt.X("tanggal:T", title="Tanggal"),
                    y=alt.Y("harga:Q", title="Harga"),
                    tooltip=["tanggal:T", "harga:Q", "predicted_label:N"],
                )
                st.altair_chart(line, use_container_width=True)
            else:
                st.info("Kolom `harga` tidak ada, chart tren harga tidak ditampilkan.")

        with right:
            dist = out["predicted_label"].value_counts().reset_index()
            dist.columns = ["Label", "Count"]
            bar = alt.Chart(dist).mark_bar().encode(
                x=alt.X("Count:Q", title="Jumlah"),
                y=alt.Y("Label:N", sort="-x", title="Prediksi"),
                tooltip=["Label", "Count"],
            )
            st.altair_chart(bar, use_container_width=True)

        st.subheader("Hasil prediksi (tabel)")
        st.dataframe(out.sort_values("tanggal"), use_container_width=True, height=520)

        st.download_button(
            "⬇️ Download hasil rentang (CSV)",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="prediction_range.csv",
            mime="text/csv",
        )

    # ----------------------------
    # Mode 3: Manual input
    # ----------------------------
    elif mode == "Input Manual":
        st.subheader("🧮 Input Manual Fitur")
        st.caption(
            "Isi nilai fitur secara manual. Untuk **komoditas/provinsi**, pilih label (akan otomatis di-encode). "
            "Pastikan nilai fitur sesuai skala yang dipakai saat training."
        )

        # Prefill defaults dari dataset yang sedang terbaca (kalau ada), biar user nggak mulai dari nol
        defaults: Dict[str, float] = {}
        for c in feat_list:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                v = df[c].replace([np.inf, -np.inf], np.nan).dropna()
                if len(v) > 0:
                    defaults[c] = float(v.median())

        # Form input
        with st.form("manual_form"):
            left, right = st.columns([1.1, 1])

            # --- Categorical & time features (lebih nyaman pakai pilihan)
            with left:
                st.markdown("**Kategori & Waktu**")

                manual_values: Dict[str, float] = {}

                # Komoditas
                if "komoditas" in feat_list:
                    kom_classes = [str(x).lower() for x in getattr(kom_enc, "classes_", [])]
                    kom_in = st.selectbox("Komoditas", kom_classes, index=0 if kom_classes else None)
                    if kom_in is not None:
                        manual_values["komoditas"] = encode_label_to_id(kom_in, kom_enc)

                # Provinsi
                if "provinsi" in feat_list:
                    prov_classes = [str(x).lower() for x in getattr(prov_enc, "classes_", [])]
                    prov_in = st.selectbox("Provinsi", prov_classes, index=0 if prov_classes else None)
                    if prov_in is not None:
                        manual_values["provinsi"] = encode_label_to_id(prov_in, prov_enc)

                # Optional helper: pilih tanggal lalu auto-isi fitur waktu
                use_date_helper = st.toggle("Gunakan helper tanggal (auto isi fitur waktu)", value=True)
                picked_date = None
                if use_date_helper:
                    picked_date = st.date_input("Tanggal (opsional)")
                    dts = pd.Timestamp(picked_date)
                    # isi fitur waktu yang ada di feat_list
                    if "year" in feat_list:
                        manual_values["year"] = int(dts.year)
                    if "month" in feat_list:
                        manual_values["month"] = int(dts.month)
                    if "day" in feat_list:
                        manual_values["day"] = int(dts.day)
                    if "week" in feat_list:
                        # week of year
                        manual_values["week"] = int(dts.isocalendar().week)
                    if "day_of_week" in feat_list:
                        manual_values["day_of_week"] = int(dts.dayofweek)

                # Kalau helper off, tampilkan input manual untuk fitur waktu
                if not use_date_helper:
                    if "year" in feat_list:
                        manual_values["year"] = st.number_input("year", value=int(defaults.get("year", 2023)), step=1)
                    if "month" in feat_list:
                        manual_values["month"] = st.number_input("month", value=int(defaults.get("month", 1)), min_value=1, max_value=12, step=1)
                    if "week" in feat_list:
                        manual_values["week"] = st.number_input("week", value=int(defaults.get("week", 1)), min_value=1, max_value=53, step=1)
                    if "day" in feat_list:
                        manual_values["day"] = st.number_input("day", value=int(defaults.get("day", 1)), min_value=1, max_value=31, step=1)
                    if "day_of_week" in feat_list:
                        manual_values["day_of_week"] = st.number_input("day_of_week", value=int(defaults.get("day_of_week", 0)), min_value=0, max_value=6, step=1)

            # --- Numeric feature groups
            with right:
                st.markdown("**Fitur Numerik**")

                # Group columns
                lag_cols = [c for c in feat_list if c.lower().startswith("lag_") or "rolling" in c.lower()]
                ext_cols = [c for c in feat_list if c.lower().startswith("kurs_") or c.lower().startswith("global_") or "google_trend" in c.lower() or c.lower().startswith("trend_")]
                other_cols = [c for c in feat_list if c not in lag_cols + ext_cols + ["komoditas", "provinsi", "year", "month", "week", "day", "day_of_week"]]

                with st.expander("Lag & Rolling", expanded=True):
                    for c in lag_cols:
                        manual_values[c] = st.number_input(
                            c,
                            value=float(defaults.get(c, 0.0)),
                            step=1.0,
                            format="%.6f",
                        )

                with st.expander("Eksternal (Kurs/Global/Trend)", expanded=True):
                    for c in ext_cols:
                        manual_values[c] = st.number_input(
                            c,
                            value=float(defaults.get(c, 0.0)),
                            step=0.1,
                            format="%.6f",
                        )

                if other_cols:
                    with st.expander("Lainnya", expanded=False):
                        for c in other_cols:
                            manual_values[c] = st.number_input(
                                c,
                                value=float(defaults.get(c, 0.0)),
                                step=1.0,
                                format="%.6f",
                            )

            submitted = st.form_submit_button("Prediksi")

        if not submitted:
            st.stop()

        # Build one-row dataframe with the exact feature order
        row_dict = {c: manual_values.get(c, defaults.get(c, 0.0)) for c in feat_list}
        one = pd.DataFrame([row_dict])

        # Predict
        try:
            X = build_X(one, feat_list, scaler)
        except Exception as e:
            st.error(f"Gagal membangun X untuk prediksi manual: {e}")
            st.stop()

        pred, proba = predict_with_proba(model, X)
        pred_id = int(pred[0])
        try:
            pred_label = y_enc.inverse_transform([pred_id])[0]
        except Exception:
            pred_label = str(pred_id)

        confidence = None
        proba_df = None
        if proba is not None:
            probs = proba[0]
            confidence = float(np.max(probs))
            try:
                labels = list(y_enc.inverse_transform(np.arange(len(probs))))
            except Exception:
                labels = [str(i) for i in range(len(probs))]
            proba_df = (
                pd.DataFrame({"Label": labels, "Probabilitas": probs})
                .sort_values("Probabilitas", ascending=False)
                .reset_index(drop=True)
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            _kpi("Prediksi Level", f"<span class='pill'>{pred_label}</span>", "")
        with c2:
            _kpi("Class ID", str(pred_id), "")
        with c3:
            _kpi("Confidence", _fmt_float(confidence, 3) if confidence is not None else "-", "max probability")

        t1, t2 = st.tabs(["🎯 Probabilitas", "🧾 Input fitur"]) 
        with t1:
            if proba_df is None:
                st.info("Model ini tidak menyediakan `predict_proba`. (Masih bisa prediksi label.)")
            else:
                chart = (
                    alt.Chart(proba_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("Probabilitas:Q", title="Probabilitas"),
                        y=alt.Y("Label:N", sort="-x", title="Kelas"),
                        tooltip=["Label", alt.Tooltip("Probabilitas:Q", format=".4f")],
                    )
                )
                st.altair_chart(chart, use_container_width=True)
                st.dataframe(proba_df, use_container_width=True)

        with t2:
            view = one.T.reset_index()
            view.columns = ["Fitur", "Nilai"]
            st.dataframe(view, use_container_width=True, height=520)

        out = one.copy()
        out["predicted_class_id"] = pred_id
        out["predicted_label"] = pred_label
        if confidence is not None:
            out["confidence"] = confidence

        st.download_button(
            "⬇️ Download hasil (CSV)",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="prediction_manual.csv",
            mime="text/csv",
        )

    # ----------------------------
    # Mode 4: Batch upload
    # ----------------------------
    else:
        st.info(
            "Batch mode: upload CSV yang **sudah berisi kolom fitur lengkap** sesuai `feature_list.pkl`. "
            "(Misal hasil export feature engineering.)"
        )

        batch = st.file_uploader("Upload CSV fitur", type=["csv"])
        if batch is None:
            st.stop()

        try:
            bdf = pd.read_csv(batch)
        except Exception as e:
            st.error(f"Gagal membaca CSV: {e}")
            st.stop()

        try:
            X = build_X(bdf, feat_list, scaler)
        except Exception as e:
            st.error(f"CSV belum sesuai format fitur: {e}")
            st.stop()

        pred, proba = predict_with_proba(model, X)
        pred = pred.astype(int)
        try:
            labels = y_enc.inverse_transform(pred)
        except Exception:
            labels = pred.astype(str)

        out = bdf.copy()
        out["predicted_class_id"] = pred
        out["predicted_label"] = labels

        if proba is not None:
            out["confidence"] = np.max(proba, axis=1)
            # also add per-class probability columns for transparency
            try:
                class_labels = list(y_enc.inverse_transform(np.arange(proba.shape[1])))
            except Exception:
                class_labels = [f"class_{i}" for i in range(proba.shape[1])]
            for i, lab in enumerate(class_labels):
                out[f"proba_{lab}"] = proba[:, i]

        st.subheader("Hasil")
        st.dataframe(out.head(200), use_container_width=True, height=520)

        st.download_button(
            "⬇️ Download output (CSV)",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="prediction_batch.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
