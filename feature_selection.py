"""
feature_selection.py — Bài 2: Feature Selection dựa trên Correlation cho Wine Dataset
- Tính ma trận Correlation giữa các đặc trưng đầu vào
- Thử nghiệm các tập feature được lựa chọn khác nhau với thuật toán Linear Regression
- So sánh hiệu năng qua độ đo Mean Absolute Error (MAE)
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ──────────────────────────────────────────
# 1) Load data
# ──────────────────────────────────────────
csv_path = "data/wine.csv"
df = pd.read_csv(csv_path)

# features và target cho bài 2 (chúng ta thử nghiệm dự đoán thuộc tính liên tục như proline bằng Linear Regression)
# hoặc dự đoán thuộc tính class bằng Linear Regression (như một bài toán hồi quy).
# Ở đây ta sẽ lấy thuộc tính 'proline' hoặc 'alcohol' làm biến mục tiêu hồi quy cho Linear Regression.
# Hãy chọn 'alcohol' làm biến target hồi quy.
target_col = "alcohol"
X = df.drop(columns=[target_col])
y = df[target_col]

# ──────────────────────────────────────────
# 2) Correlation matrix
# ──────────────────────────────────────────
corr_matrix = df.corr()

# Save correlation heatmap as HTML
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

# Correlation với target (alcohol)
corr_with_target = corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)

print(f"\nCorrelation with target '{target_col}' (absolute value):")
print(corr_with_target.to_string())

# ──────────────────────────────────────────
# 3) Define feature subsets with different thresholds
# ──────────────────────────────────────────
thresholds = [0.0, 0.1, 0.2, 0.3, 0.4]
feature_sets = {}

for thresh in thresholds:
    selected = corr_with_target[corr_with_target >= thresh].index.tolist()
    if len(selected) > 0:
        feature_sets[f"corr>={thresh}"] = selected

# Top-3, Top-5
top_features = corr_with_target.index.tolist()
feature_sets["Top-3"] = top_features[:3]
feature_sets["Top-5"] = top_features[:5]
feature_sets["All features"] = top_features

# ──────────────────────────────────────────
# 4) Train LinearRegression with each feature subset and compare MAE
# ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

results = []

for subset_name, feature_list in feature_sets.items():
    X_tr = X_train[feature_list]
    X_te = X_test[feature_list]
    
    # Scale
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)
    
    # Train
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
    
    print(f"{subset_name:20s} | {len(feature_list):2d} features | MAE = {mae:.4f}")

# ──────────────────────────────────────────
# 5) Save results
# ──────────────────────────────────────────
df_results = pd.DataFrame(results)
csv_out = "results/feature_selection.csv"
df_results.to_csv(csv_out, index=False)

print(f"\n{'='*60}")
print(f"Feature selection results saved -> {csv_out}")
print(f"{'='*60}")
print(df_results.to_string(index=False))
