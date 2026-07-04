import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from model_definitions import CustomLDAClassifier

# -----------------------------
# Model loading (cached)
# -----------------------------
@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


# -----------------------------
# Discover available models
# -----------------------------
def _get_available_models():
    models = {}
    results_dir = "results"

    if os.path.isdir(results_dir):
        for f in os.listdir(results_dir):
            if f.endswith(".joblib") and f != "scaler.joblib":
                name = f.replace(".joblib", "")
                models[name] = os.path.join(results_dir, f)

    if not models and os.path.exists("model.ckpt"):
        models["BestModel"] = "model.ckpt"

    return models


# -----------------------------
# Page
# -----------------------------
def Home():
    st.markdown("## 🍷 Wine Origin Classifier")
    st.caption("Nhập các thông số hoá học của rượu để dự đoán nhóm xuất xứ của giống nho (Class 1, 2, hoặc 3).")

    # ── Model selector ──
    available_models = _get_available_models()

    if not available_models:
        st.error("Không tìm thấy mô hình nào. Hãy chạy `python train.py` trước!")
        st.stop()

    model_names = list(available_models.keys())
    selected_model = st.selectbox(
        "🧠 Chọn mô hình phân loại",
        model_names,
        index=0,
        key="home_model_selector",
    )

    st.divider()

    st.markdown(
        """
        <div style="padding: 18px; border: 1px solid rgba(255,255,255,0.12);
                    border-radius: 14px; background: rgba(255,255,255,0.03); margin-bottom: 10px;">
        <h3 style="margin: 0 0 10px 0;">Thông số hoá học đầu vào</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 13 features for Wine dataset
    c1, c2, c3 = st.columns(3)

    with c1:
        alcohol = st.number_input("Alcohol (Nồng độ cồn)", min_value=10.0, max_value=16.0, value=13.0, step=0.1)
        malic_acid = st.number_input("Malic Acid (Axit malic)", min_value=0.0, max_value=7.0, value=2.3, step=0.1)
        ash = st.number_input("Ash (Tro)", min_value=1.0, max_value=4.0, value=2.3, step=0.1)
        alcalinity_of_ash = st.number_input("Alcalinity of Ash (Độ kiềm của tro)", min_value=5.0, max_value=35.0, value=19.0, step=0.5)
        magnesium = st.number_input("Magnesium (Magie)", min_value=50.0, max_value=200.0, value=100.0, step=1.0)

    with c2:
        total_phenols = st.number_input("Total Phenols (Tổng lượng Phenol)", min_value=0.5, max_value=5.0, value=2.3, step=0.1)
        flavanoids = st.number_input("Flavanoids (Chất chống oxy hóa Flavanoid)", min_value=0.0, max_value=6.0, value=2.0, step=0.1)
        nonflavanoid_phenols = st.number_input("Nonflavanoid Phenols (Phenol không phải Flavanoid)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        proanthocyanins = st.number_input("Proanthocyanins (Chất màu Proanthocyanin)", min_value=0.0, max_value=5.0, value=1.6, step=0.1)

    with c3:
        color_intensity = st.number_input("Color Intensity (Cường độ màu)", min_value=1.0, max_value=15.0, value=5.0, step=0.1)
        hue = st.number_input("Hue (Tông màu)", min_value=0.1, max_value=2.0, value=1.0, step=0.05)
        od280_od315 = st.number_input("OD280/OD315 of diluted wines (Độ loãng của rượu)", min_value=1.0, max_value=5.0, value=2.6, step=0.1)
        proline = st.number_input("Proline (Axit amin Proline)", min_value=100.0, max_value=2000.0, value=750.0, step=10.0)

    st.divider()

    predict_clicked = st.button("🔮 Dự đoán xuất xứ", key="btn_predict", type="primary")

    st.divider()

    st.markdown(
        """
        <div style="padding: 18px; border: 1px solid rgba(255,255,255,0.12);
                    border-radius: 14px; background: rgba(255,255,255,0.03); margin-bottom: 10px;">
        <h3 style="margin: 0 0 10px 0;">Kết quả phân loại</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if predict_clicked:
        # 13 features in correct order as in train.py (excluding class)
        X_input = pd.DataFrame(
            [
                {
                    "alcohol": float(alcohol),
                    "malic_acid": float(malic_acid),
                    "ash": float(ash),
                    "alcalinity_of_ash": float(alcalinity_of_ash),
                    "magnesium": float(magnesium),
                    "total_phenols": float(total_phenols),
                    "flavanoids": float(flavanoids),
                    "nonflavanoid_phenols": float(nonflavanoid_phenols),
                    "proanthocyanins": float(proanthocyanins),
                    "color_intensity": float(color_intensity),
                    "hue": float(hue),
                    "od280_od315": float(od280_od315),
                    "proline": float(proline),
                }
            ]
        )

        try:
            # Load scaler and transform input
            scaler_path = "results/scaler.joblib"
            if not os.path.exists(scaler_path):
                st.error("Không tìm thấy scaler.joblib. Hãy chạy `python train.py` trước!")
                st.stop()
                
            scaler = joblib.load(scaler_path)
            X_input_scaled = scaler.transform(X_input)

            model_path = available_models[selected_model]
            model = load_model(model_path)
            
            # Predict
            pred = model.predict(X_input_scaled)[0]

            # Custom response decoration
            st.metric(
                label=f"Phân loại dự đoán ({selected_model})",
                value=f"Class {int(pred)}",
                help="Loại rượu (cultivar 1, 2 hoặc 3)",
            )

        except FileNotFoundError:
            st.error("Không tìm thấy mô hình hoặc scaler.")
        except Exception as e:
            st.error(f"Dự đoán thất bại: {e}")
    else:
        st.caption("Click **Dự đoán xuất xứ** để phân loại rượu.")
