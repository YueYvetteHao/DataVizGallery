"""
Run this script to generate xgboost_artifacts.pkl.
Requires: pandas, numpy, xgboost, scikit-learn, scipy

    python export_model.py

Outputs: xgboost_artifacts.pkl
"""
import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_val_predict, RandomizedSearchCV,
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

DIR = os.path.dirname(os.path.abspath(__file__))

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
merged = pd.read_csv(os.path.join(DIR, "glioma_gdsc_ccle_merged.csv"))
top_500_genes = pd.read_csv(os.path.join(DIR, "top_500_variable_genes_expression.csv"))

drug_target = ("Staurosporine", "Sepantronium bromide")

# ── 2. Build Decision labels ──────────────────────────────────────────────────
decision_map = (
    merged[merged["DRUG_NAME"].isin(drug_target)]
    .dropna(subset=["Z_SCORE"])
    .groupby("CELL_LINE_NAME")
    .apply(lambda g: g.loc[g["Z_SCORE"].idxmin(), "DRUG_NAME"])
)
merged["Decision"] = merged["CELL_LINE_NAME"].map(decision_map)

# ── 3. Build feature matrix ───────────────────────────────────────────────────
gene_desc_map = top_500_genes.set_index("Name")["Description"].to_dict()
expr = top_500_genes.set_index("Name").drop(columns=["Description"], errors="ignore")
X_all = expr.T.copy()
X_all.index.name = "CELL_LINE_NAME_STRIPPED"

decision_per_cl = (
    merged.dropna(subset=["Decision"])
    .drop_duplicates(subset=["CELL_LINE_NAME"])
    [["CELL_LINE_NAME", "Decision"]]
    .copy()
)
decision_per_cl["CELL_LINE_NAME_STRIPPED"] = (
    decision_per_cl["CELL_LINE_NAME"].str.replace("-", "", regex=False)
)
decision_per_cl = decision_per_cl.set_index("CELL_LINE_NAME_STRIPPED")
df_model = X_all.join(decision_per_cl[["Decision"]], how="inner")

# ── 4. Filter rare classes ────────────────────────────────────────────────────
class_counts = df_model["Decision"].value_counts()
valid_classes = class_counts[class_counts >= 5].index
df_model = df_model[df_model["Decision"].isin(valid_classes)]
print(f"Dataset: {df_model.shape[0]} samples × {df_model.shape[1]-1} features")
print(df_model["Decision"].value_counts().to_string())

X = df_model.drop(columns=["Decision"]).values.astype(float)
y_raw = df_model["Decision"].values
le = LabelEncoder()
y = le.fit_transform(y_raw)
gene_names = df_model.columns[:-1].tolist()
n_classes = len(le.classes_)

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def _top20_importance(importances):
    top_idx = np.argsort(importances)[-20:][::-1]
    return [
        {
            "gene": gene_names[i],
            "description": gene_desc_map.get(gene_names[i], gene_names[i]),
            "importance": float(importances[i]),
        }
        for i in top_idx
    ]


# ── 5. Base XGBoost ───────────────────────────────────────────────────────────
print("\nBase XGBoost...")
base_xgb = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="mlogloss", random_state=42, n_jobs=-1,
)
base_xgb_cv = cross_val_score(base_xgb, X, y, cv=cv5, scoring="accuracy")
base_xgb_pred = cross_val_predict(base_xgb, X, y, cv=cv5)
base_xgb_cm = confusion_matrix(y, base_xgb_pred, labels=np.arange(n_classes))
base_xgb_report = classification_report(
    y, base_xgb_pred, target_names=le.classes_, zero_division=0, output_dict=True,
)
base_xgb.fit(X, y)
print(f"Base XGBoost: {base_xgb_cv.mean():.3f} ± {base_xgb_cv.std():.3f}")

# ── 6. Tuned XGBoost (RandomizedSearchCV) ────────────────────────────────────
print("\nTuned XGBoost (RandomizedSearchCV, n_iter=3000)...")
xgb_param_dist = {
    "n_estimators":     randint(100, 600),
    "max_depth":        randint(2, 8),
    "learning_rate":    uniform(0.01, 0.2),
    "subsample":        uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.5, 0.5),
    "min_child_weight": randint(1, 10),
    "gamma":            uniform(0, 0.5),
}
xgb_search = RandomizedSearchCV(
    estimator=xgb.XGBClassifier(eval_metric="mlogloss", random_state=42, n_jobs=-1),
    param_distributions=xgb_param_dist,
    n_iter=3000,
    scoring="accuracy",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    random_state=42,
    n_jobs=-1,
    verbose=1,
    refit=True,
)
xgb_search.fit(X, y)
tuned_xgb = xgb_search.best_estimator_
print(f"Best CV accuracy: {xgb_search.best_score_:.3f}")
print(f"Best params: {xgb_search.best_params_}")

tuned_xgb_cv = cross_val_score(tuned_xgb, X, y, cv=cv5, scoring="accuracy")
tuned_xgb_pred = cross_val_predict(tuned_xgb, X, y, cv=cv5)
tuned_xgb_cm = confusion_matrix(y, tuned_xgb_pred, labels=np.arange(n_classes))
tuned_xgb_report = classification_report(
    y, tuned_xgb_pred, target_names=le.classes_, zero_division=0, output_dict=True,
)
print(f"Tuned XGBoost: {tuned_xgb_cv.mean():.3f} ± {tuned_xgb_cv.std():.3f}")

# ── 7. Base Random Forest ─────────────────────────────────────────────────────
print("\nBase Random Forest...")
base_rf = RandomForestClassifier(
    n_estimators=500, max_depth=None, min_samples_leaf=1, random_state=42, n_jobs=-1,
)
base_rf_cv = cross_val_score(base_rf, X, y, cv=cv5, scoring="accuracy")
base_rf_pred = cross_val_predict(base_rf, X, y, cv=cv5)
base_rf_cm = confusion_matrix(y, base_rf_pred, labels=np.arange(n_classes))
base_rf_report = classification_report(
    y, base_rf_pred, target_names=le.classes_, zero_division=0, output_dict=True,
)
base_rf.fit(X, y)
print(f"Base Random Forest: {base_rf_cv.mean():.3f} ± {base_rf_cv.std():.3f}")

# ── 8. Tuned Random Forest (RandomizedSearchCV) ───────────────────────────────
print("\nTuned Random Forest (RandomizedSearchCV, n_iter=100)...")
rf_param_dist = {
    "n_estimators":      randint(100, 800),
    "max_depth":         [None, 5, 10, 20, 30],
    "min_samples_split": randint(2, 20),
    "min_samples_leaf":  randint(1, 10),
    "max_features":      ["sqrt", "log2", 0.3, 0.5],
}
rf_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=rf_param_dist,
    n_iter=100,
    scoring="accuracy",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    random_state=42,
    n_jobs=-1,
    verbose=1,
    refit=True,
)
rf_search.fit(X, y)
tuned_rf = rf_search.best_estimator_
print(f"Best CV accuracy: {rf_search.best_score_:.3f}")
print(f"Best params: {rf_search.best_params_}")

tuned_rf_cv = cross_val_score(tuned_rf, X, y, cv=cv5, scoring="accuracy")
tuned_rf_pred = cross_val_predict(tuned_rf, X, y, cv=cv5)
tuned_rf_cm = confusion_matrix(y, tuned_rf_pred, labels=np.arange(n_classes))
tuned_rf_report = classification_report(
    y, tuned_rf_pred, target_names=le.classes_, zero_division=0, output_dict=True,
)
print(f"Tuned Random Forest: {tuned_rf_cv.mean():.3f} ± {tuned_rf_cv.std():.3f}")

# ── 9. ROC curves (OOF predict_proba, all 4 models) ──────────────────────────
print("\nROC curves...")
y_bin = label_binarize(y, classes=np.arange(n_classes))
if y_bin.shape[1] == 1:
    y_bin = np.hstack([1 - y_bin, y_bin])

roc_data = {}
for mname, mobj in [
    ("Base XGBoost", base_xgb),
    ("Tuned XGBoost", tuned_xgb),
    ("Base Random Forest", base_rf),
    ("Tuned Random Forest", tuned_rf),
]:
    y_proba = cross_val_predict(
        mobj, X, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        method="predict_proba",
    )
    roc_data[mname] = {}
    for cls_idx, cls_name in enumerate(le.classes_):
        fpr, tpr, _ = roc_curve(y_bin[:, cls_idx], y_proba[:, cls_idx])
        roc_data[mname][cls_name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(auc(fpr, tpr)),
        }

# ── 10. Save ──────────────────────────────────────────────────────────────────
artifacts = {
    "task": "Drug decision prediction for glioma cell lines",
    "drugs": list(drug_target),
    "class_labels": le.classes_.tolist(),
    "n_samples": int(df_model.shape[0]),
    "n_features": int(df_model.shape[1] - 1),
    # Base XGBoost
    "base_xgb_params": {k: v for k, v in base_xgb.get_params().items() if v is not None},
    "base_xgb_cv_scores": base_xgb_cv.tolist(),
    "base_xgb_confusion_matrix": base_xgb_cm.tolist(),
    "base_xgb_classification_report": base_xgb_report,
    "base_xgb_feature_importance_top20": _top20_importance(base_xgb.feature_importances_),
    # Tuned XGBoost
    "tuned_xgb_best_params": xgb_search.best_params_,
    "tuned_xgb_best_score": float(xgb_search.best_score_),
    "tuned_xgb_cv_scores": tuned_xgb_cv.tolist(),
    "tuned_xgb_confusion_matrix": tuned_xgb_cm.tolist(),
    "tuned_xgb_classification_report": tuned_xgb_report,
    "tuned_xgb_feature_importance_top20": _top20_importance(tuned_xgb.feature_importances_),
    # Base Random Forest
    "base_rf_params": {k: v for k, v in base_rf.get_params().items() if v is not None},
    "base_rf_cv_scores": base_rf_cv.tolist(),
    "base_rf_confusion_matrix": base_rf_cm.tolist(),
    "base_rf_classification_report": base_rf_report,
    "base_rf_feature_importance_top20": _top20_importance(base_rf.feature_importances_),
    # Tuned Random Forest
    "tuned_rf_best_params": rf_search.best_params_,
    "tuned_rf_best_score": float(rf_search.best_score_),
    "tuned_rf_cv_scores": tuned_rf_cv.tolist(),
    "tuned_rf_confusion_matrix": tuned_rf_cm.tolist(),
    "tuned_rf_classification_report": tuned_rf_report,
    "tuned_rf_feature_importance_top20": _top20_importance(tuned_rf.feature_importances_),
    # ROC curves
    "roc_data": roc_data,
}

out_path = os.path.join(DIR, "xgboost_artifacts.pkl")
with open(out_path, "wb") as f:
    pickle.dump(artifacts, f)

size_kb = os.path.getsize(out_path) / 1024
print(f"\nSaved: {out_path}")
print(f"Size: {size_kb:.1f} KB")
