import sys
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import joblib

import streamlit as st

try:
    import plotly.express as px
except Exception:  # pragma: no cover
    px = None


def _patch_numpy_private_module_paths() -> None:
    """Make old/new NumPy private module paths loadable when unpickling.

    Some joblib/pickle artifacts may reference `numpy._core.*` which may not exist
    on older NumPy builds. This patch aliases the modules so unpickling succeeds.
    """
    try:
        import numpy as _np

        sys.modules.setdefault("numpy._core", _np.core)

        import numpy.core._multiarray_umath as _mau
        sys.modules.setdefault("numpy._core._multiarray_umath", _mau)
    except Exception:
        pass


@st.cache_resource(show_spinner=False)
def load_artifacts():
    _patch_numpy_private_module_paths()

    base = Path(__file__).parent

    model = joblib.load(base / "best_model.pkl")
    feature_list = joblib.load(base / "feature_list.pkl")
    komoditas_encoder = joblib.load(base / "komoditas_encoder.pkl")
    provinsi_encoder = joblib.load(base / "provinsi_encoder.pkl")
    scaler = joblib.load(base / "scaler.pkl")
    target_encoder = joblib.load(base / "target_encoder.pkl")

    metadata_path = base / "metadata.pkl"
    metadata = {}
    if metadata_path.exists():
        try:
            metadata = joblib.load(metadata_path)
        except Exception:
            metadata = {}

    return {
        "model": model,
        "feature_list": list(feature_list),
        "komoditas_encoder": komoditas_encoder,
        "provinsi_encoder": provinsi_encoder,
        "scaler": scaler,
        "target_encoder": target_encoder,
        "metadata": metadata,
    }


def _safe_encode(label_encoder, value: str, feature_name: str) -> int:
    value_norm = (value or "").strip().lower()
    classes = set(map(str, label_encoder.classes_))
    if value_norm not in classes:
        raise ValueError(
            f"Nilai '{value}' untuk '{feature_name}' tidak dikenal oleh encoder. "
            f"Gunakan salah satu dari: {', '.join(list(label_encoder.classes_)[:15])}..."
        )
    return int(label_encoder.transform([value_norm])[0])


def _date_features(d: date) -> dict:
    iso = d.isocalendar()
    return {
        "month": int(d.month),
        "week": int(iso.week),
        "day": int(d.day),
        "day_of_week": int(d.weekday()),
    }


def build_feature_row(
    artifacts: dict,
    tanggal: date,
    komoditas: str,
    provinsi: str,
    numeric_inputs: dict,
    use_defaults_for_missing: bool = True,
) -> pd.DataFrame:
    feature_list = artifacts["feature_list"]
    scaler = artifacts["scaler"]

    row = {}
    row["komoditas"] = _safe_encode(artifacts["komoditas_encoder"], komoditas, "komoditas")
    row["provinsi"] = _safe_encode(artifacts["provinsi_encoder"], provinsi, "provinsi")

    row.update(_date_features(tanggal))

    scaler_features = list(getattr(scaler, "feature_names_in_", []))
    if not scaler_features:
        raise RuntimeError("Scaler tidak memiliki feature_names_in_.")

    defaults = {f: float(m) for f, m in zip(scaler_features, scaler.mean_)}

    unscaled_values = []
    for f in scaler_features:
        if f in numeric_inputs and numeric_inputs[f] is not None:
            unscaled_values.append(float(numeric_inputs[f]))
        elif use_defaults_for_missing:
            unscaled_values.append(defaults[f])
        else:
            raise ValueError(f"Fitur '{f}' belum diisi.")

    scaled = scaler.transform([unscaled_values])[0]
    for f, v in zip(scaler_features, scaled):
        row[f] = float(v)

    missing = [c for c in feature_list if c not in row]
    if missing:
        raise ValueError(
            "Ada fitur yang belum terbentuk: " + ", ".join(missing) +
            ". Pastikan feature_list.pkl sesuai dengan pipeline training."
        )

    return pd.DataFrame([[row[c] for c in feature_list]], columns=feature_list)


def predict_one(artifacts: dict, X: pd.DataFrame):
    model = artifacts["model"]
    target_encoder = artifacts["target_encoder"]

    proba = model.predict_proba(X)[0]
    class_ids = model.classes_

    labels = target_encoder.inverse_transform(class_ids.astype(int))
    pred_id = int(class_ids[int(np.argmax(proba))])
    pred_label = str(target_encoder.inverse_transform([pred_id])[0])

    proba_map = {str(lbl): float(p) for lbl, p in zip(labels, proba)}
    return pred_label, proba_map


def _inject_css():
    st.markdown(
        """
<style>
.block-container { padding-top: 1.3rem; padding-bottom: 2.0rem; }
.hero {
  padding: 18px 18px;
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(18, 24, 38, 0.95), rgba(31, 41, 55, 0.90));
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}
.hero h1 { margin: 0; font-size: 1.55rem; }
.hero p { margin: 6px 0 0; opacity: 0.85; }
.metric-card {
  border-radius: 16px;
  border: 1px solid rgba(0,0,0,0.08);
  background: rgba(255,255,255,0.65);
  padding: 14px 14px;
}
hr { border: 0; height: 1px; background: rgba(0,0,0,0.06); margin: 16px 0; }
</style>
        """,
        unsafe_allow_html=True,
    )


def _proba_chart(proba_map: dict):
    if px is None:
        st.info("Plotly belum terpasang. Tambahkan 'plotly' ke requirements.txt untuk chart.")
        st.write(proba_map)
        return

    dfp = pd.DataFrame({"Kelas": list(proba_map.keys()), "Probabilitas": list(proba_map.values())})
    dfp = dfp.sort_values("Probabilitas", ascending=False)
    fig = px.bar(dfp, x="Kelas", y="Probabilitas", text="Probabilitas")
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20), yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)


def _feature_importance_chart(artifacts: dict):
    model = artifacts["model"]
    feats = artifacts["feature_list"]

    if not hasattr(model, "feature_importances_"):
        st.info("Model ini tidak menyediakan feature_importances_.")
        return

    imp = pd.DataFrame({"Fitur": feats, "Importance": model.feature_importances_})
    imp = imp.sort_values("Importance", ascending=False)

    if px is None:
        st.dataframe(imp, use_container_width=True)
        return

    fig = px.bar(imp.head(15), x="Importance", y="Fitur", orientation="h")
    fig.update_layout(height=520, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)


def _template_df(artifacts: dict, n_rows: int = 3) -> pd.DataFrame:
    feat = artifacts["feature_list"]
    scaler_feats = list(artifacts["scaler"].feature_names_in_)

    df = pd.DataFrame({c: [None] * n_rows for c in feat})

    df["komoditas"] = ["beras medium"] * n_rows
    df["provinsi"] = ["dki jakarta"] * n_rows

    today = date.today()
    df["month"] = [today.month] * n_rows
    df["week"] = [today.isocalendar().week] * n_rows
    df["day"] = [today.day] * n_rows
    df["day_of_week"] = [today.weekday()] * n_rows

    means = artifacts["scaler"].mean_
    for c, m in zip(scaler_feats, means):
        if c in df.columns:
            df[c] = [float(m)] * n_rows

    return df


def _preprocess_batch_df(artifacts: dict, df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    for date_col in ["tanggal", "date", "Date", "Tanggal"]:
        if date_col in df.columns:
            d = pd.to_datetime(df[date_col], errors="coerce")
            df["month"] = d.dt.month
            df["week"] = d.dt.isocalendar().week.astype("Int64")
            df["day"] = d.dt.day
            df["day_of_week"] = d.dt.dayofweek
            break

    ke = artifacts["komoditas_encoder"]
    pe = artifacts["provinsi_encoder"]

    if df["komoditas"].dtype == object:
        df["komoditas"] = df["komoditas"].astype(str).str.strip().str.lower().map(
            lambda x: _safe_encode(ke, x, "komoditas")
        )

    if df["provinsi"].dtype == object:
        df["provinsi"] = df["provinsi"].astype(str).str.strip().str.lower().map(
            lambda x: _safe_encode(pe, x, "provinsi")
        )

    scaler = artifacts["scaler"]
    scaler_feats = list(scaler.feature_names_in_)

    for c in scaler_feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    scaled = scaler.transform(df[scaler_feats].to_numpy())
    for i, c in enumerate(scaler_feats):
        df[c] = scaled[:, i]

    df = df[artifacts["feature_list"]]
    return df


def main():
    st.set_page_config(page_title="Prediksi Level Harga Komoditas", page_icon="📈", layout="wide")
    _inject_css()

    artifacts = load_artifacts()

    st.markdown(
        """
<div class="hero">
  <h1>📈 Dashboard Prediksi Level Harga Komoditas</h1>
  <p>Mode sederhana untuk input cepat, dan mode advanced untuk memasukkan indikator makro (kurs & harga komoditas global).</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    with st.sidebar:
        st.header("⚙️ Pengaturan")
        mode = st.radio(
            "Mode input",
            ["Default (Sederhana)", "Advanced (Lengkap)"],
            help="Default: input inti. Advanced: tambah kurs, komoditas global, dan Google Trend.",
        )
        st.divider()
        st.caption("Artifacts yang harus ada di folder app: best_model.pkl, feature_list.pkl, scaler.pkl, *encoder*.pkl")

    tab_pred, tab_batch, tab_model = st.tabs(["🎯 Prediksi", "📦 Batch", "🧠 Model"])

    with tab_pred:
        col_in, col_out = st.columns([1.05, 1.0], gap="large")

        with col_in:
            st.subheader("Input")

            komoditas_opts = list(artifacts["komoditas_encoder"].classes_)
            provinsi_opts = list(artifacts["provinsi_encoder"].classes_)

            defaults = {f: float(m) for f, m in zip(artifacts["scaler"].feature_names_in_, artifacts["scaler"].mean_)}

            with st.form("predict_form", clear_on_submit=False):
                tanggal = st.date_input("Tanggal", value=date.today())

                c1, c2 = st.columns(2)
                with c1:
                    komoditas = st.selectbox("Komoditas", komoditas_opts, index=komoditas_opts.index("beras medium") if "beras medium" in komoditas_opts else 0)
                with c2:
                    provinsi = st.selectbox("Provinsi", provinsi_opts, index=provinsi_opts.index("dki jakarta") if "dki jakarta" in provinsi_opts else 0)

                st.markdown("**Harga historis (wajib)**")
                l1, l2, l3 = st.columns(3)
                with l1:
                    harga_lag_1 = st.number_input("Harga kemarin (lag 1)", min_value=0.0, value=float(defaults["harga_lag_1"]), step=100.0)
                with l2:
                    harga_lag_7 = st.number_input("Harga 7 hari lalu (lag 7)", min_value=0.0, value=float(defaults["harga_lag_7"]), step=100.0)
                with l3:
                    harga_lag_30 = st.number_input("Harga 30 hari lalu (lag 30)", min_value=0.0, value=float(defaults["harga_lag_30"]), step=100.0)

                numeric_inputs = {
                    "harga_lag_1": harga_lag_1,
                    "harga_lag_7": harga_lag_7,
                    "harga_lag_30": harga_lag_30,
                }

                if mode.startswith("Advanced"):
                    st.markdown("---")
                    st.markdown("**Indikator makro (opsional)**")

                    with st.expander("💱 Kurs mata uang", expanded=True):
                        k1, k2 = st.columns(2)
                        with k1:
                            numeric_inputs["kurs_usdidr"] = st.number_input("USD/IDR", value=float(defaults["kurs_usdidr"]), step=10.0)
                            numeric_inputs["kurs_sgdusd"] = st.number_input("SGD/USD", value=float(defaults["kurs_sgdusd"]), step=0.01, format="%.4f")
                        with k2:
                            numeric_inputs["kurs_myrusd"] = st.number_input("MYR/USD", value=float(defaults["kurs_myrusd"]), step=0.01, format="%.4f")
                            numeric_inputs["kurs_thbusd"] = st.number_input("THB/USD", value=float(defaults["kurs_thbusd"]), step=0.01, format="%.4f")

                    with st.expander("🌍 Komoditas global", expanded=False):
                        g1, g2, g3 = st.columns(3)
                        with g1:
                            numeric_inputs["global_crude_oil"] = st.number_input("Crude Oil", value=float(defaults["global_crude_oil"]), step=1.0)
                            numeric_inputs["global_natural_gas"] = st.number_input("Natural Gas", value=float(defaults["global_natural_gas"]), step=0.1)
                        with g2:
                            numeric_inputs["global_coal"] = st.number_input("Coal", value=float(defaults["global_coal"]), step=1.0)
                            numeric_inputs["global_palm_oil"] = st.number_input("Palm Oil", value=float(defaults["global_palm_oil"]), step=1.0)
                        with g3:
                            numeric_inputs["global_sugar"] = st.number_input("Sugar", value=float(defaults["global_sugar"]), step=1.0)
                            numeric_inputs["global_wheat"] = st.number_input("Wheat", value=float(defaults["global_wheat"]), step=1.0)

                    with st.expander("🔎 Google Trend", expanded=False):
                        numeric_inputs["google_trend"] = st.number_input("Google Trend Index", min_value=0.0, value=float(defaults["google_trend"]), step=1.0)

                submitted = st.form_submit_button("🚀 Prediksi")

        with col_out:
            st.subheader("Hasil")

            if not submitted:
                st.info("Isi input di kiri, lalu klik **Prediksi**.")
            else:
                try:
                    X = build_feature_row(
                        artifacts=artifacts,
                        tanggal=tanggal,
                        komoditas=komoditas,
                        provinsi=provinsi,
                        numeric_inputs=numeric_inputs,
                        use_defaults_for_missing=True,
                    )
                    pred_label, proba_map = predict_one(artifacts, X)

                    cA, cB = st.columns([0.9, 1.1])
                    with cA:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Prediksi", pred_label)
                        st.markdown("</div>", unsafe_allow_html=True)
                    with cB:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Key insight", "Kelas dengan probabilitas tertinggi")
                        st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown("### Probabilitas")
                    _proba_chart(proba_map)

                    with st.expander("Lihat fitur yang dipakai model (setelah preprocessing)", expanded=False):
                        st.dataframe(X, use_container_width=True)

                    st.download_button(
                        "⬇️ Download fitur (CSV)",
                        X.to_csv(index=False).encode("utf-8"),
                        file_name="features_single_row.csv",
                        mime="text/csv",
                    )

                except Exception as e:
                    st.error(f"Gagal melakukan prediksi: {e}")

    with tab_batch:
        st.subheader("Batch Prediction")
        st.caption(
            "Upload CSV berisi kolom sesuai feature_list. Jika ada kolom 'tanggal', "
            "dashboard akan otomatis membentuk month/week/day/day_of_week."
        )

        c1, c2 = st.columns([1.1, 0.9], gap="large")

        with c1:
            up = st.file_uploader("Upload CSV", type=["csv"])
            if up is not None:
                df_raw = pd.read_csv(up)
            else:
                df_raw = _template_df(artifacts, n_rows=5)

            st.markdown("**Edit langsung (opsional)**")
            df_edit = st.data_editor(df_raw, use_container_width=True, num_rows="dynamic")

            st.download_button(
                "⬇️ Download template CSV",
                _template_df(artifacts, n_rows=10).to_csv(index=False).encode("utf-8"),
                file_name="template_input.csv",
                mime="text/csv",
            )

        with c2:
            st.markdown("#### Jalankan")
            run = st.button("▶️ Prediksi Batch", use_container_width=True)

            if run:
                try:
                    df_proc = _preprocess_batch_df(artifacts, df_edit)
                    model = artifacts["model"]
                    target_encoder = artifacts["target_encoder"]

                    proba = model.predict_proba(df_proc)
                    pred_id = model.predict(df_proc).astype(int)
                    pred_label = target_encoder.inverse_transform(pred_id)
                    labels = target_encoder.inverse_transform(model.classes_.astype(int))

                    out = df_edit.copy()
                    out["prediksi"] = pred_label
                    for i, lbl in enumerate(labels):
                        out[f"proba_{lbl}"] = proba[:, i]

                    st.success(f"Selesai: {len(out)} baris diprediksi.")
                    st.dataframe(out, use_container_width=True)

                    st.download_button(
                        "⬇️ Download hasil (CSV)",
                        out.to_csv(index=False).encode("utf-8"),
                        file_name="prediksi_batch.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                except Exception as e:
                    st.error(f"Gagal batch predict: {e}")

    with tab_model:
        st.subheader("Tentang Model")

        meta = artifacts.get("metadata", {}) or {}

        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.markdown("**Ringkasan**")
            st.write(
                {
                    "Tipe": meta.get("task_type", "classification"),
                    "Target": meta.get("target", "(unknown)"),
                    "Kelas": list(artifacts["target_encoder"].classes_),
                    "Jumlah fitur (model)": getattr(artifacts["model"], "n_features_in_", len(artifacts["feature_list"])),
                }
            )

            if meta:
                st.markdown("**Data split (metadata)**")
                st.write({"Train": meta.get("train_period"), "Test": meta.get("test_period")})

            st.markdown("**Fitur yang dipakai model**")
            st.code("\n".join(artifacts["feature_list"]))

        with c2:
            st.markdown("**Feature Importance (Top 15)**")
            _feature_importance_chart(artifacts)

        st.info(
            "Catatan: input indikator (kurs, komoditas global, Google Trend) "
            "akan dinormalisasi (StandardScaler) sebelum masuk model."
        )


if __name__ == "__main__":
    main()
