"""
FeatureSelection.py — Bài 2: Trang Streamlit Feature Selection cho Wine Dataset (Target: Alcohol)
Hiển thị Correlation heatmap, so sánh MAE với các tập feature khác nhau
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def FeatureSelection() -> None:
    st.header("🔬 Feature Selection — Bài 2")
    st.caption(
        "Lựa chọn đặc trưng dựa trên Correlation và so sánh hiệu suất với Linear Regression để dự đoán nồng độ cồn (Alcohol)"
    )

    # ─── Load data ───
    data_path = "data/wine.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"Không tìm thấy file: `{data_path}`")
        st.stop()

    target_col = "alcohol"

    st.divider()

    # ═══════════════════════════════════════════
    # PHẦN A: Correlation — Đồ thị minh hoạ
    # ═══════════════════════════════════════════
    st.subheader("📊 A. Phương pháp dựa trên Correlation")

    # Correlation matrix heatmap
    corr_matrix = df.corr()

    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=".2f",
        title="Ma trận Correlation — Wine Dataset",
        aspect="auto",
        color_continuous_scale="RdBu_r",
    )
    fig_heatmap.update_layout(width=800, height=750)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.divider()

    # Correlation with target
    corr_with_target = (
        corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
    )

    st.markdown("#### Mức độ tương quan của từng đặc trưng với `alcohol`")

    fig_bar = px.bar(
        x=corr_with_target.index,
        y=corr_with_target.values,
        title=f"Correlation với {target_col} (|r|)",
        labels={"x": "Feature", "y": "|Correlation|"},
        color=corr_with_target.values,
        color_continuous_scale="Viridis",
    )
    fig_bar.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Show correlation table
    corr_df = pd.DataFrame(
        {
            "Feature": corr_with_target.index,
            "|Correlation|": corr_with_target.values.round(4),
        }
    )
    st.dataframe(corr_df, use_container_width=True, hide_index=True)

    st.divider()

    # ═══════════════════════════════════════════
    # PHẦN B: Thử nghiệm tập feature — Linear Regression — MAE
    # ═══════════════════════════════════════════
    st.subheader("📈 B. Thử nghiệm với Linear Regression — So sánh MAE")

    # Check for pre-computed results
    csv_path = "results/feature_selection.csv"
    if os.path.exists(csv_path):
        df_results = pd.read_csv(csv_path)
        st.success("Đã tải kết quả từ `results/feature_selection.csv`")
    else:
        st.info("Đang tính toán kết quả Feature Selection...")

        # Prepare data
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Define feature subsets
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4]
        feature_sets = {}
        top_features = corr_with_target.index.tolist()

        for thresh in thresholds:
            selected = corr_with_target[corr_with_target >= thresh].index.tolist()
            if selected:
                feature_sets[f"corr>={thresh}"] = selected

        feature_sets["Top-3"] = top_features[:3]
        feature_sets["Top-5"] = top_features[:5]
        feature_sets["All features"] = top_features

        results = []
        for subset_name, feat_list in feature_sets.items():
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train[feat_list])
            X_te = scaler.transform(X_test[feat_list])

            model = LinearRegression()
            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)
            mae = mean_absolute_error(y_test, y_pred)

            results.append(
                {
                    "Feature_Subset": subset_name,
                    "Num_Features": len(feat_list),
                    "Features": ", ".join(feat_list),
                    "MAE": round(mae, 4),
                }
            )

        df_results = pd.DataFrame(results)

    # ─── Display results ───
    st.markdown("#### Bảng so sánh MAE (Hồi quy nồng độ cồn)")
    st.dataframe(
        df_results[["Feature_Subset", "Num_Features", "MAE"]],
        use_container_width=True,
        hide_index=True,
    )

    # Bar chart MAE comparison
    fig_mae = px.bar(
        df_results,
        x="Feature_Subset",
        y="MAE",
        color="MAE",
        title="So sánh MAE của mô hình Linear Regression",
        text_auto=".4f",
        color_continuous_scale="Reds_r",
    )
    fig_mae.update_layout(xaxis_tickangle=-45, showlegend=False)
    st.plotly_chart(fig_mae, use_container_width=True)

    st.divider()

    # ─── Detailed feature list per subset ───
    with st.expander("📋 Chi tiết các tập Feature"):
        for _, row in df_results.iterrows():
            st.markdown(
                f"**{row['Feature_Subset']}** ({row['Num_Features']} features, MAE={row['MAE']:.4f})"
            )
            st.caption(row.get("Features", "N/A"))

    # ─── Conclusion ───
    st.divider()
    best_row = df_results.loc[df_results["MAE"].idxmin()]
    st.success(
        f"🏆 Tập feature tốt nhất: **{best_row['Feature_Subset']}** "
        f"với {best_row['Num_Features']} features — MAE = {best_row['MAE']:.4f}"
    )
