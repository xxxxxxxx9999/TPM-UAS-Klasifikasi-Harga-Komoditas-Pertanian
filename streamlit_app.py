
import os
import sys
from pathlib import Path
from datetime import datetime, date

import numpy as np
import pandas as pd
import joblib
import streamlit as st

try:
    import plotly.express as px
except Exception:  # pragma: no cover
    px = None


# ============================================================
# 0) Compatibility patch (pickle artifacts may reference numpy._core)
# ============================================================
def _patch_numpy_private_module_paths() -> None:
    """
    Some joblib/pickle artifacts may reference `numpy._core.*`.
    On many NumPy 1.x builds, that path doesn't exist. We alias it to keep
    unpickling working across environments.
    """
    try:
        import numpy as _np

        sys.modules.setdefault("numpy._core", _np.core)
        import numpy.core._multiarray_umath as _mau
        sys.modules.setdefault("numpy._core._multiarray_umath", _mau)
    except Exception:
        pass


# ============================================================
# 1) Page config + premium CSS
# ============================================================
st.set_page_config(
    page_title="Prediksi Harga Komoditas",
    page_icon="📈",
    layout="wide"
)

st.markdown(
    """
<style>
/* Layout */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
.hero {
  padding: 22px 22px;
  border-radius: 20px;
  background: linear-gradient(135deg, rgba(17, 24, 39, 0.96), rgba(31, 41, 55, 0.90));
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow: 0 14px 35px rgba(0,0,0,0.25);
  margin-bottom: 18px;
}
.hero h1 { margin: 0; font-size: 1.6rem; color: #F9FAFB; }
.hero p  { margin: 6px 0 0; opacity: 0.85; color: #E5E7EB; }

.main-card {
  padding: 26px;
  border-radius: 20px;
  background-color: #ffffff;
  box-shadow: 0 10px 25px rgba(0,0,0,0.06);
  border: 1px solid #f0f2f6;
}
.badge {
  display: inline-block;
  padding: 5px 10px;
  border-radius: 999px;
  font-weight: 700;
  font-size: 0.85rem;
}
.kpi-title { font-size: 1.0rem; opacity: 0.7; margin: 0; text-align: center; }
.kpi-value { font-size: 3.3rem; font-weight: 900; margin: 0; text-align: center; line-height: 1.05; }

.stButton>button {
  width: 100%;
  border-radius: 12px;
  height: 3.0em;
  font-weight: 800;
}
.small-note { color: #6B7280; font-size: 0.88rem; }
hr { border: 0; height: 1px; background: rgba(0,0,0,0.06); margin: 16px 0; }
</style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 2) Artifact loading (supports /artifacts folder or repo root)
# ============================================================
REQUIRED_FILES = [
    "best_model.pkl",
    "scaler.pkl",
    "komoditas_encoder.pkl",
    "provinsi_encoder.pkl",
    "target_encoder.pkl",
    "feature_list.pkl",
    # metadata.pkl is optional
]


def _find_artifact_dir() -> Path:
    """
    Streamlit Cloud biasanya menaruh repo di /mount/src/<repo_name>.
    Banyak orang menyimpan pkl di folder `artifacts/`.
    Fungsi ini akan mencari otomatis.
    """
    env_dir = os.environ.get("ARTIFACT_DIR", "").strip()
    candidates = []

    if env_dir:
        candidates.append(Path(env_dir))

    here = Path(__file__).parent
    candidates += [
        here / "artifacts",
        here,
        Path.cwd() / "artifacts",
        Path.cwd(),
    ]

    checked = []
    for d in candidates:
        d = d.resolve()
        if d in checked:
            continue
        checked.append(d)
        ok = all((d / f).exists() for f in REQUIRED_FILES)
        if ok:
            return d

    # If none match, raise a helpful error
    msg_lines = ["Artifacts tidak ditemukan.", "", "Folder yang dicek:"]
    for d in checked:
        try:
            items = sorted([p.name for p in d.iterdir()])[:30]
            msg_lines.append(f"- {d}  (isi: {items}{' ...' if len(items)==30 else ''})")
        except Exception:
            msg_lines.append(f"- {d}  (tidak bisa membaca isi folder)")

    msg_lines += [
        "",
        "Pastikan file .pkl berada di salah satu folder di atas, contoh struktur repo:",
        "  /streamlit_app.py",
        "  /requirements.txt",
        "  /artifacts/best_model.pkl",
        "  /artifacts/scaler.pkl",
        "  /artifacts/komoditas_encoder.pkl",
        "  /artifacts/provinsi_encoder.pkl",
        "  /artifacts/target_encoder.pkl",
        "  /artifacts/feature_list.pkl",
    ]
    raise FileNotFoundError("\n".join(msg_lines))


@st.cache_resource(show_spinner=True)
def load_assets():
    _patch_numpy_private_module_paths()

    art_dir = _find_artifact_dir()

    assets = {
        "artifact_dir": art_dir,
        "model": joblib.load(art_dir / "best_model.pkl"),
        "scaler": joblib.load(art_dir / "scaler.pkl"),
        "le_komo": joblib.load(art_dir / "komoditas_encoder.pkl"),
        "le_prov": joblib.load(art_dir / "provinsi_encoder.pkl"),
        "target_enc": joblib.load(art_dir / "target_encoder.pkl"),
        "feat_list": list(joblib.load(art_dir / "feature_list.pkl")),
    }

    meta_path = art_dir / "metadata.pkl"
    if meta_path.exists():
        try:
            assets["metadata"] = joblib.load(meta_path)
        except Exception:
            assets["metadata"] = {}
    else:
        assets["metadata"] = {}

    return assets


# ============================================================
# 3) Helpers (feature build + predict)
# ============================================================
def _date_features(d: date) -> dict:
    iso = d.isocalendar()
    return {
        "month": int(d.month),
        "week": int(iso.week),
        "day": int(d.day),
        "day_of_week": int(d.weekday()),
    }


def _safe_encode(le, value: str, feature_name: str) -> int:
    v = (value or "").strip().lower()
    classes = set(map(str, le.classes_))
    if v not in classes:
        raise ValueError(
            f"Nilai '{value}' untuk '{feature_name}' tidak dikenal. "
            f"Contoh nilai valid: {', '.join(list(le.classes_)[:12])} ..."
        )
    return int(le.transform([v])[0])


def build_single_row(
    assets: dict,
    sel_date: date,
    sel_komo: str,
    sel_prov: str,
    raw_numeric: dict,
) -> pd.DataFrame:
    """
    Membentuk satu baris fitur sesuai `feature_list.pkl`.
    Semua fitur numerik akan di-scale memakai StandardScaler.
    """
    feat_list = assets["feat_list"]
    scaler = assets["scaler"]

    # Encode kategori
    row = {
        "komoditas": _safe_encode(assets["le_komo"], sel_komo, "komoditas"),
        "provinsi": _safe_encode(assets["le_prov"], sel_prov, "provinsi"),
    }
    row.update(_date_features(sel_date))

    # Scale numerik sesuai urutan scaler.feature_names_in_
    scaler_feats = list(getattr(scaler, "feature_names_in_", []))
    if not scaler_feats:
        raise RuntimeError("Scaler tidak punya feature_names_in_. Tidak bisa memastikan urutan fitur.")

    defaults = {f: float(m) for f, m in zip(scaler_feats, scaler.mean_)}
    unscaled = []
    for f in scaler_feats:
        val = raw_numeric.get(f, None)
        if val is None:
            # fallback ke mean training (paling aman untuk mode standard)
            val = defaults[f]
        unscaled.append(float(val))

    scaled = scaler.transform([unscaled])[0]
    for f, v in zip(scaler_feats, scaled):
        row[f] = float(v)

    # Pastikan lengkap sesuai feature_list
    missing = [c for c in feat_list if c not in row]
    if missing:
        raise ValueError(
            "Ada fitur yang belum terbentuk: " + ", ".join(missing) +
            ". Cek kesesuaian feature_list.pkl vs pipeline training."
        )

    X = pd.DataFrame([[row[c] for c in feat_list]], columns=feat_list)
    return X


def predict_with_proba(assets: dict, X: pd.DataFrame):
    model = assets["model"]
    target_enc = assets["target_enc"]

    proba = model.predict_proba(X)[0]
    class_ids = model.classes_.astype(int)
    labels = target_enc.inverse_transform(class_ids)

    pred_id = int(class_ids[int(np.argmax(proba))])
    pred_label = str(target_enc.inverse_transform([pred_id])[0])

    proba_map = {str(lbl): float(p) for lbl, p in zip(labels, proba)}
    return pred_label, proba_map


def _plot_proba(proba_map: dict):
    if px is None:
        st.write(proba_map)
        return
    dfp = pd.DataFrame({"Level": list(proba_map.keys()), "Prob": list(proba_map.values())})
    dfp = dfp.sort_values("Prob", ascending=False)
    fig = px.bar(dfp, x="Level", y="Prob", text="Prob")
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=10, b=20), yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 4) App UI
# ============================================================
st.markdown(
    """
<div class="hero">
  <h1>📈 Prediksi Level Harga Komoditas</h1>
  <p>Mode <b>Standard</b> untuk input cepat (tanggal + harga), dan mode <b>Advanced</b> untuk memasukkan indikator kurs & global.</p>
</div>
    """,
    unsafe_allow_html=True
)

# Load assets with friendly error on Cloud
try:
    assets = load_assets()
except Exception as e:
    st.error("Gagal memuat artifacts (model/scaler/encoder).")
    st.code(str(e))
    st.stop()

with st.sidebar:
    st.header("⚙️ Pengaturan")
    mode = st.radio(
        "Mode Input",
        ["Standard (Cepat & Lokal)", "Advanced (Full Parameter)"],
        horizontal=False
    )
    st.divider()
    st.caption("Lokasi artifacts yang terdeteksi:")
    st.code(str(assets["artifact_dir"]))
    with st.expander("🔎 Debug (cek file di artifacts)", expanded=False):
        try:
            items = sorted([p.name for p in Path(assets["artifact_dir"]).iterdir()])
            st.write(items)
        except Exception as _:
            st.write("Tidak bisa membaca isi folder artifacts.")


tab_pred, tab_batch, tab_about = st.tabs(["🎯 Prediksi", "📦 Batch", "🧠 About"])

# ---------------------------
# Tab 1: Single prediction
# ---------------------------
with tab_pred:
    col_left, col_right = st.columns([1, 1.4], gap="large")

    # Defaults (training mean) for macro features
    scaler_feats = list(assets["scaler"].feature_names_in_)
    scaler_means = list(assets["scaler"].mean_)
    defaults = {f: float(m) for f, m in zip(scaler_feats, scaler_means)}

    with col_left:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.subheader("📍 Data Utama")

        sel_komo = st.selectbox("Komoditas", list(assets["le_komo"].classes_), index=list(assets["le_komo"].classes_).index("beras medium") if "beras medium" in assets["le_komo"].classes_ else 0)
        sel_prov = st.selectbox("Provinsi", list(assets["le_prov"].classes_), index=list(assets["le_prov"].classes_).index("dki jakarta") if "dki jakarta" in assets["le_prov"].classes_ else 0)
        sel_date = st.date_input("Tanggal Prediksi", value=date.today())

        st.subheader("💰 Harga Historis")
        h_l1 = st.number_input("Harga Kemarin (H-1)", min_value=0.0, value=float(defaults.get("harga_lag_1", 15000.0)), step=500.0)

        raw_numeric = {}
        # Standard: auto-estimate lag 7 & lag 30
        if mode.startswith("Standard"):
            st.info("ℹ️ Mode Standard mengisi H-7, H-30, kurs, dan indikator global secara otomatis (nilai rata-rata training).")
            # heuristic (boleh kamu ubah)
            h_l7 = max(0.0, float(h_l1) - 200.0)
            h_l30 = max(0.0, float(h_l1) - 500.0)
        else:
            h_l7 = st.number_input("Harga Minggu Lalu (H-7)", min_value=0.0, value=float(defaults.get("harga_lag_7", h_l1)), step=500.0)
            h_l30 = st.number_input("Harga Bulan Lalu (H-30)", min_value=0.0, value=float(defaults.get("harga_lag_30", h_l1)), step=500.0)

        raw_numeric["harga_lag_1"] = h_l1
        raw_numeric["harga_lag_7"] = h_l7
        raw_numeric["harga_lag_30"] = h_l30

        if mode.startswith("Advanced"):
            st.subheader("🌐 Indikator Ekonomi & Global")
            c1, c2 = st.columns(2)
            with c1:
                raw_numeric["kurs_usdidr"] = st.number_input("USD/IDR", value=float(defaults.get("kurs_usdidr", 15500.0)), step=10.0)
                raw_numeric["kurs_myrusd"] = st.number_input("MYR/USD", value=float(defaults.get("kurs_myrusd", 0.22)), format="%.4f")
                raw_numeric["kurs_sgdusd"] = st.number_input("SGD/USD", value=float(defaults.get("kurs_sgdusd", 0.74)), format="%.4f")
                raw_numeric["kurs_thbusd"] = st.number_input("THB/USD", value=float(defaults.get("kurs_thbusd", 0.028)), format="%.5f")
                raw_numeric["google_trend"] = st.number_input("Google Trend", value=float(defaults.get("google_trend", 50.0)), step=1.0)
            with c2:
                raw_numeric["global_crude_oil"] = st.number_input("Crude Oil", value=float(defaults.get("global_crude_oil", 80.0)), step=1.0)
                raw_numeric["global_natural_gas"] = st.number_input("Natural Gas", value=float(defaults.get("global_natural_gas", 2.5)), step=0.1)
                raw_numeric["global_coal"] = st.number_input("Coal", value=float(defaults.get("global_coal", 130.0)), step=1.0)
                raw_numeric["global_palm_oil"] = st.number_input("Palm Oil", value=float(defaults.get("global_palm_oil", 900.0)), step=1.0)
                raw_numeric["global_sugar"] = st.number_input("Sugar", value=float(defaults.get("global_sugar", 0.20)), step=0.1)
                raw_numeric["global_wheat"] = st.number_input("Wheat", value=float(defaults.get("global_wheat", 600.0)), step=10.0)

        st.markdown("<hr/>", unsafe_allow_html=True)
        run = st.button("🚀 CEK PREDIKSI HARGA", type="primary")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.subheader("✅ Hasil Prediksi")

        if not run:
            st.info("Klik tombol **CEK PREDIKSI HARGA** untuk melihat hasil.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            try:
                X = build_single_row(
                    assets=assets,
                    sel_date=sel_date,
                    sel_komo=sel_komo,
                    sel_prov=sel_prov,
                    raw_numeric=raw_numeric,
                )
                pred_label, proba_map = predict_with_proba(assets, X)

                # Color mapping
                colors = {
                    "Tinggi": ("#FF4B4B", "#FFF5F5"),
                    "Sedang": ("#F59E0B", "#FFFBEB"),
                    "Rendah": ("#10B981", "#ECFDF5"),
                }
                main_color, bg_color = colors.get(pred_label, ("#111827", "#F3F4F6"))

                st.markdown(
                    f"""
<div class="main-card" style="background-color: {bg_color}; border-top: 10px solid {main_color}; margin-top: 10px;">
  <p class="kpi-title">Estimasi Tingkat Harga</p>
  <p class="kpi-value" style="color: {main_color};">{pred_label.upper()}</p>
  <p style="text-align:center; color:#555; margin-top: 12px;">
    Komoditas: <b>{sel_komo}</b> | Provinsi: <b>{sel_prov}</b> | Tanggal: <b>{sel_date.strftime('%d %B %Y')}</b>
  </p>
</div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("#### Probabilitas")
                _plot_proba(proba_map)

                with st.expander("🔍 Lihat fitur setelah preprocessing (untuk debugging)", expanded=False):
                    st.dataframe(X, use_container_width=True)
                    st.download_button(
                        "⬇️ Download fitur (CSV)",
                        X.to_csv(index=False).encode("utf-8"),
                        file_name="features_single_row.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"Prediksi gagal: {e}")

            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<p style='text-align:center; color:#9CA3AF; font-size:0.82rem; margin-top:16px;'>© Prediksi Harga Komoditas • Streamlit App</p>",
        unsafe_allow_html=True
    )


# ---------------------------
# Tab 2: Batch prediction
# ---------------------------
with tab_batch:
    st.subheader("📦 Batch Prediction (CSV)")

    st.caption(
        "Upload CSV dengan kolom sesuai `feature_list.pkl` atau gunakan template. "
        "Kalau ada kolom `tanggal`, sistem otomatis membentuk month/week/day/day_of_week."
    )

    feat_list = assets["feat_list"]
    scaler_feats = list(assets["scaler"].feature_names_in_)
    means = assets["scaler"].mean_

    def make_template(n=5):
        df = pd.DataFrame({c: [None] * n for c in feat_list})
        df["komoditas"] = ["beras medium"] * n
        df["provinsi"] = ["dki jakarta"] * n

        today = date.today()
        df["month"] = [today.month] * n
        df["week"] = [today.isocalendar().week] * n
        df["day"] = [today.day] * n
        df["day_of_week"] = [today.weekday()] * n

        for c, m in zip(scaler_feats, means):
            if c in df.columns:
                df[c] = [float(m)] * n
        return df

    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is None:
        df_raw = make_template(8)
    else:
        df_raw = pd.read_csv(up)

    df_edit = st.data_editor(df_raw, use_container_width=True, num_rows="dynamic")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.download_button(
            "⬇️ Download template CSV",
            make_template(20).to_csv(index=False).encode("utf-8"),
            file_name="template_batch.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col_b:
        run_batch = st.button("▶️ Prediksi Batch", use_container_width=True)

    def preprocess_batch(df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in.copy()

        # date -> derived
        for date_col in ["tanggal", "Tanggal", "date", "Date"]:
            if date_col in df.columns:
                d = pd.to_datetime(df[date_col], errors="coerce")
                df["month"] = d.dt.month
                df["week"] = d.dt.isocalendar().week.astype("Int64")
                df["day"] = d.dt.day
                df["day_of_week"] = d.dt.dayofweek
                break

        # encode categories if needed
        if "komoditas" in df.columns and df["komoditas"].dtype == object:
            df["komoditas"] = df["komoditas"].astype(str).str.strip().str.lower().map(
                lambda x: _safe_encode(assets["le_komo"], x, "komoditas")
            )
        if "provinsi" in df.columns and df["provinsi"].dtype == object:
            df["provinsi"] = df["provinsi"].astype(str).str.strip().str.lower().map(
                lambda x: _safe_encode(assets["le_prov"], x, "provinsi")
            )

        # scale numeric
        for c in scaler_feats:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        scaled = assets["scaler"].transform(df[scaler_feats].to_numpy())
        for i, c in enumerate(scaler_feats):
            df[c] = scaled[:, i]

        return df[feat_list]

    if run_batch:
        try:
            df_proc = preprocess_batch(df_edit)
            model = assets["model"]
            target_enc = assets["target_enc"]

            proba = model.predict_proba(df_proc)
            pred_id = model.predict(df_proc).astype(int)
            pred_label = target_enc.inverse_transform(pred_id)

            labels = target_enc.inverse_transform(model.classes_.astype(int))

            out = df_edit.copy()
            out["prediksi"] = pred_label
            for i, lbl in enumerate(labels):
                out[f"proba_{lbl}"] = proba[:, i]

            st.success(f"Selesai: {len(out)} baris diprediksi.")
            st.dataframe(out, use_container_width=True)

            st.download_button(
                "⬇️ Download hasil (CSV)",
                out.to_csv(index=False).encode("utf-8"),
                file_name="hasil_prediksi_batch.csv",
                mime="text/csv",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Batch prediksi gagal: {e}")


# ---------------------------
# Tab 3: About / Model info
# ---------------------------
with tab_about:
    st.subheader("🧠 Informasi Model")

    meta = assets.get("metadata", {}) or {}
    st.write({
        "Task": meta.get("task_type", "classification"),
        "Target": meta.get("target", "price_level"),
        "Kelas target": list(assets["target_enc"].classes_),
        "Jumlah fitur input": len(assets["feat_list"]),
        "Artifacts folder": str(assets["artifact_dir"]),
    })

    with st.expander("Daftar fitur (feature_list.pkl)", expanded=False):
        st.code("\n".join(assets["feat_list"]))

    if hasattr(assets["model"], "feature_importances_"):
        st.markdown("#### Feature Importance (Top 12)")
        imp = pd.DataFrame({
            "Fitur": assets["feat_list"],
            "Importance": assets["model"].feature_importances_
        }).sort_values("Importance", ascending=False).head(12)

        if px is None:
            st.dataframe(imp, use_container_width=True)
        else:
            fig = px.bar(imp, x="Importance", y="Fitur", orientation="h")
            fig.update_layout(height=520, margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(fig, use_container_width=True)
