"""
feature_selection.py — Bài 2: Feature Selection dựa trên Correlation
Sử dụng sklearn.feature_selection (SelectKBest, f_regression)
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression

# ──────────────────────────────────────────
# 1) Load data
# ──────────────────────────────────────────
csv_path = "data/wine.csv"
df = pd.read_csv(csv_path)

target_col = "alcohol"
X = df.drop(columns=[target_col])
y = df[target_col]

# ──────────────────────────────────────────
# 2) Correlation matrix (Giữ lại để vẽ Heatmap theo yêu cầu đề)
# ──────────────────────────────────────────
corr_matrix = df.corr()
fig = px.imshow(
    corr_matrix,
    text_auto=".2f",
    title="Correlation Matrix — Wine Dataset",
    aspect="auto",
    color_continuous_scale="RdBu_r",
)
fig.update_layout(width=900, height=800)
os.makedirs("results", exist_ok=True)
fig.write_html("results/correlation_heatmap.html")
print("Saved correlation heatmap -> results/correlation_heatmap.html")

# ──────────────────────────────────────────
# 3) Sklearn Feature Selection & Evaluate
# ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

results = []
k_values = [3, 5, 8, 10, X.shape[1]]

for k in k_values:
    subset_name = f"Top-{k} (SelectKBest)" if k != X.shape[1] else "All features"
    
    # Sử dụng SelectKBest với f_regression (đo lường tương quan tuyến tính)
    selector = SelectKBest(score_func=f_regression, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Lấy tên các đặc trưng được chọn
    selected_mask = selector.get_support()
    feature_list = X.columns[selected_mask].tolist()
    
    # Scale
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train_selected)
    X_te_scaled = scaler.transform(X_test_selected)
    
    # Train Linear Regression
    model = LinearRegression()
    model.fit(X_tr_scaled, y_train)
    
    # Predict & Evaluate
    y_pred = model.predict(X_te_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    
    results.append({
        "Feature_Subset": subset_name,
        "Num_Features": len(feature_list),
        "Features": ", ".join(feature_list),
        "MAE": round(mae, 4)
    })
    
    print(f"{subset_name:25s} | {len(feature_list):2d} features | MAE = {mae:.4f}")

# ──────────────────────────────────────────
# 4) Save results
# ──────────────────────────────────────────
df_results = pd.DataFrame(results)
csv_out = "results/feature_selection.csv"
df_results.to_csv(csv_out, index=False)

print(f"\n{'='*60}")
print(f"Feature selection results saved -> {csv_out}")
print(f"{'='*60}")
print(df_results.to_string(index=False))
