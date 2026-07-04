"""
train.py — Bài 1: Phân loại rượu (Wine Classification) trên Wine dataset
Các mô hình: LDA (tự code từ Midterm), DecisionTree, RandomForest, LogisticRegression
So sánh: Accuracy, Precision, Recall, F1-score (weighted)
Đo thời gian training & testing
"""

import os
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from model_definitions import CustomLDAClassifier

# ──────────────────────────────────────────
# 1) Load data
# ──────────────────────────────────────────
csv_path = "data/wine.csv"
df = pd.read_csv(csv_path)
print(f"Dataset shape: {df.shape}")
print(f"Classes distribution:\n{df['class'].value_counts()}\n")

# Target và Features
target_col = "class"
X = df.drop(columns=[target_col])
y = df[target_col]

# ──────────────────────────────────────────
# 2) Train / test split (70/30 - stratify)
# ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}\n")

# ──────────────────────────────────────────
# 3) Define Pipeline Wrapper for custom LDA
# ──────────────────────────────────────────
# Chúng ta sẽ scale dữ liệu trước khi train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "CustomLDA": CustomLDAClassifier(n_components=2),
    "LogisticRegression": LogisticRegression(random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
}

# ──────────────────────────────────────────
# 4) Train & Evaluate
# ──────────────────────────────────────────
results = []
os.makedirs("results", exist_ok=True)

for name, clf in models.items():
    print(f"{'='*50}")
    print(f"Training: {name}")
    print(f"{'='*50}")

    t0 = time.perf_counter()
    if name == "CustomLDA":
        clf.fit(X_train_scaled, y_train.values)
    else:
        clf.fit(X_train_scaled, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = clf.predict(X_test_scaled)
    test_time = time.perf_counter() - t0

    # Tính metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Train time: {train_time:.4f}s")
    print(f"  Test  time: {test_time:.4f}s\n")
    print("  --- Chi tiết từng class ---")
    print(classification_report(y_test, y_pred))

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1_Score": round(f1, 4),
        "Train_Time(s)": round(train_time, 4),
        "Test_Time(s)": round(test_time, 4),
    })

    # Save model
    model_path = f"results/{name}.joblib"
    joblib.dump(clf, model_path)
    print(f"  Saved -> {model_path}")

# Lưu scaler
joblib.dump(scaler, "results/scaler.joblib")

# ──────────────────────────────────────────
# 5) Save comparison table
# ──────────────────────────────────────────
df_results = pd.DataFrame(results)
csv_out = "results/model_comparison.csv"
df_results.to_csv(csv_out, index=False)

print(f"\n{'='*50}")
print(f"Model comparison saved -> {csv_out}")
print(f"{'='*50}")
print(df_results.to_string(index=False))

# model.ckpt
best_model_name = df_results.loc[df_results["F1_Score"].idxmax(), "Model"]
best_clf = joblib.load(f"results/{best_model_name}.joblib")
joblib.dump(best_clf, "model.ckpt")
print(f"\nBest model ({best_model_name}) also saved as model.ckpt")
