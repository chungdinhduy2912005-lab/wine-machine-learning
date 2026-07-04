"""
ModelComparison.py — Bài 1: Trang Streamlit so sánh mô hình Phân loại rượu
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from model_definitions import CustomLDAClassifier


def ModelComparison() -> None:
    st.header("📊 So sánh mô hình — Bài 1")
    st.caption("So sánh hiệu suất của các mô hình phân loại trên Wine Dataset")

    # ─── Load comparison results ───
    csv_path = "results/model_comparison.csv"
    if not os.path.exists(csv_path):
        st.error(
            f"Chưa có file kết quả `{csv_path}`. Hãy chạy `python train.py` trước!"
        )
        st.stop()

    df_results = pd.read_csv(csv_path)

    # ─── KPIs: Best model ───
    best_idx = df_results["F1_Score"].idxmax()
    best = df_results.loc[best_idx]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🏆 Mô hình tốt nhất (F1)", best["Model"])
    with c2:
        st.metric("F1-Score (weighted)", f"{best['F1_Score']:.4f}")
    with c3:
        st.metric("Accuracy", f"{best['Accuracy']:.4f}")
    with c4:
        st.metric("Precision", f"{best['Precision']:.4f}")

    st.divider()

    # ─── Metrics table ───
    st.subheader("📋 Bảng so sánh Metrics")
    st.dataframe(
        df_results[["Model", "Accuracy", "Precision", "Recall", "F1_Score"]],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ─── Bar charts: Metrics comparison ───
    st.subheader("📈 Biểu đồ so sánh Metrics")

    tab1, tab2, tab3 = st.tabs(["Accuracy", "F1-Score", "Precision & Recall"])

    with tab1:
        fig = px.bar(
            df_results,
            x="Model",
            y="Accuracy",
            color="Model",
            title="Độ chính xác (Accuracy)",
            text_auto=".4f",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.bar(
            df_results,
            x="Model",
            y="F1_Score",
            color="Model",
            title="Weighted F1-Score",
            text_auto=".4f",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Grouped bar chart for Precision & Recall
        fig_multi = go.Figure()
        fig_multi.add_trace(
            go.Bar(
                name="Precision",
                x=df_results["Model"],
                y=df_results["Precision"],
                text=df_results["Precision"].round(4),
                textposition="auto",
            )
        )
        fig_multi.add_trace(
            go.Bar(
                name="Recall",
                x=df_results["Model"],
                y=df_results["Recall"],
                text=df_results["Recall"].round(4),
                textposition="auto",
            )
        )
        fig_multi.update_layout(
            barmode="group",
            title="So sánh Precision và Recall",
            yaxis_range=[0, 1.1],
        )
        st.plotly_chart(fig_multi, use_container_width=True)

    st.divider()

    # ─── Training & Testing time ───
    st.subheader("⏱ Thời gian Training & Testing")

    time_df = df_results[["Model", "Train_Time(s)", "Test_Time(s)"]].copy()
    st.dataframe(time_df, use_container_width=True, hide_index=True)

    fig_time = go.Figure()
    fig_time.add_trace(
        go.Bar(
            name="Train Time (s)",
            x=time_df["Model"],
            y=time_df["Train_Time(s)"],
            text=time_df["Train_Time(s)"].round(4),
            textposition="auto",
        )
    )
    fig_time.add_trace(
        go.Bar(
            name="Test Time (s)",
            x=time_df["Model"],
            y=time_df["Test_Time(s)"],
            text=time_df["Test_Time(s)"].round(4),
            textposition="auto",
        )
    )
    fig_time.update_layout(
        barmode="group",
        title="So sánh thời gian Training & Testing",
        yaxis_title="Thời gian (giây)",
    )
    st.plotly_chart(fig_time, use_container_width=True)

    st.divider()

    # ─── Confusion Matrix ───
    st.subheader("🔍 Confusion Matrix")

    # Load data and split
    data = pd.read_csv("data/wine.csv")
    X = data.drop(columns=["class"])
    y = data["class"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Scale using saved scaler
    scaler_file = "results/scaler.joblib"
    if os.path.exists(scaler_file):
        scaler = joblib.load(scaler_file)
        X_test_scaled = scaler.transform(X_test)

        model_names = df_results["Model"].tolist()
        selected_model = st.selectbox(
            "Chọn mô hình hiển thị Confusion Matrix",
            model_names,
            key="cm_model_sel",
        )

        model_file = f"results/{selected_model}.joblib"
        if os.path.exists(model_file):
            clf = joblib.load(model_file)
            y_pred = clf.predict(X_test_scaled)

            # Compute confusion matrix
            classes = sorted(np.unique(y_test))
            cm = confusion_matrix(y_test, y_pred)

            fig_cm = px.imshow(
                cm,
                x=[f"Predicted {c}" for c in classes],
                y=[f"Actual {c}" for c in classes],
                text_auto=True,
                title=f"Confusion Matrix — {selected_model}",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.warning(f"Chưa tìm thấy model file: {model_file}")
    else:
        st.warning("Chưa tìm thấy file scaler.joblib. Hãy chạy `python train.py`.")
