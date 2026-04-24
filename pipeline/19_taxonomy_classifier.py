"""
19_taxonomy_classifier.py
GAPC — Random Forest photometric taxonomy classifier.

Trains a Random Forest classifier to predict asteroid taxonomy class from
photometric and orbital features, then applies it to the full 128K-object
catalog to fill taxonomy gaps left by the GASP cross-match.

Training labels (from GASP Tier-1):
  S → "S"
  C → "C"
  X → "X"
  All others (B, D, V, K, L, T, Q, R, A, P, E) → "Other"

Features: G, H_V, a_au, ecc, inc_deg, phase_range, n_obs

Outputs:
  data/interim/taxonomy_rf_model.pkl
  data/interim/gapc_catalog_v4_step1.parquet
  plots/19_confusion_matrix.png
  plots/19_feature_importance.png
  plots/19_taxonomy_distribution.png
  logs/19_classifier_stats.txt
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  accuracy_score)
    import joblib
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("ERROR: scikit-learn and/or joblib not installed. "
          "Run: pip install scikit-learn joblib")
    sys.exit(1)

ROOT      = Path(__file__).resolve().parents[1]
CAT_PATH  = ROOT / "data" / "final"   / "gapc_catalog_v3_var.parquet"
OC_PATH   = ROOT / "data" / "interim" / "mpcorb_orbital_class.parquet"
MODEL_OUT = ROOT / "data" / "interim" / "taxonomy_rf_model.pkl"
OUT_CAT   = ROOT / "data" / "interim" / "gapc_catalog_v4_step1.parquet"
PLOT_DIR  = ROOT / "plots"
LOG_DIR   = ROOT / "logs"

FEATURES  = ["G", "H_V", "a_au", "ecc", "inc_deg", "phase_range", "n_obs"]

TAX_MAP = {
    "S": "S", "C": "C", "X": "X",
    "B": "Other", "D": "Other", "V": "Other", "K": "Other",
    "L": "Other", "T": "Other", "Q": "Other", "R": "Other",
    "A": "Other", "P": "Other", "E": "Other",
}


def map_taxonomy(raw):
    """Map raw taxonomy string to 4-group label."""
    if pd.isna(raw):
        return np.nan
    # Take first character, stripping suffixes like ':' or numeric
    t = str(raw).strip().upper()[0] if str(raw).strip() else np.nan
    return TAX_MAP.get(t, "Other")


def main():
    print("\n" + "=" * 60)
    print("  GAPC Step 19 — Taxonomy Random Forest Classifier")
    print("=" * 60)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "interim").mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading catalog: {CAT_PATH}")
    df = pd.read_parquet(CAT_PATH)
    print(f"  Catalog rows: {len(df):,}")

    print(f"Loading orbital elements: {OC_PATH}")
    oc = pd.read_parquet(OC_PATH)
    oc_cols = ["number_mp", "a_au", "ecc", "inc_deg"]
    oc = oc[[c for c in oc_cols if c in oc.columns]].drop_duplicates("number_mp")
    print(f"  Orbital rows: {len(oc):,}")

    df = df.merge(oc, on="number_mp", how="left")
    print(f"  After merge: {len(df):,} rows")

    # ── Map taxonomy to 4 groups ───────────────────────────────────────────────
    df["tax_group"] = df["gasp_taxonomy_final"].apply(map_taxonomy)

    # ── Training set ──────────────────────────────────────────────────────────
    train_mask = (
        df["tax_group"].notna() &
        df["G"].notna() &
        df["a_au"].notna()
    )
    train_full = df[train_mask].copy()
    # Drop NaN in any feature
    train_full = train_full.dropna(subset=FEATURES)
    print(f"\nTraining objects (all features present): {len(train_full):,}")
    print("  Class distribution:")
    vc = train_full["tax_group"].value_counts()
    for cls, n in vc.items():
        print(f"    {cls:>8s}: {n:,}")

    X_train = train_full[FEATURES].values
    y_train = train_full["tax_group"].values

    # ── 5-fold stratified cross-validation ────────────────────────────────────
    print("\nRunning 5-fold stratified cross-validation ...")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(clf, X_train, y_train, cv=cv, n_jobs=-1)

    acc = accuracy_score(y_train, y_pred_cv)
    report = classification_report(y_train, y_pred_cv, digits=3)
    cm = confusion_matrix(y_train, y_pred_cv,
                          labels=["C", "Other", "S", "X"])

    print(f"\n  5-fold CV accuracy: {acc:.4f}")
    print("\n" + report)

    # ── Feature importances (from a quick fit on all training data) ───────────
    print("Computing feature importances on full training set ...")
    clf_fi = RandomForestClassifier(
        n_estimators=300, max_depth=12, random_state=42, n_jobs=-1
    )
    clf_fi.fit(X_train, y_train)
    importances = pd.Series(clf_fi.feature_importances_, index=FEATURES)
    importances_sorted = importances.sort_values(ascending=False)
    print("\n  Feature importances:")
    for feat, imp in importances_sorted.items():
        print(f"    {feat:>15s}: {imp:.4f}")

    # ── Train final model on ALL training data ─────────────────────────────────
    print("\nTraining final model on full training set ...")
    clf_final = RandomForestClassifier(
        n_estimators=300, max_depth=12, random_state=42, n_jobs=-1
    )
    clf_final.fit(X_train, y_train)
    joblib.dump(clf_final, MODEL_OUT)
    print(f"  Model saved: {MODEL_OUT}")

    # ── Apply to full catalog ──────────────────────────────────────────────────
    predict_mask = df[FEATURES].notna().all(axis=1)
    print(f"\nApplying classifier to {predict_mask.sum():,} objects with all features ...")

    X_all = df.loc[predict_mask, FEATURES].values
    pred_labels = clf_final.predict(X_all)
    pred_proba = clf_final.predict_proba(X_all).max(axis=1)

    df["predicted_taxonomy"] = np.nan
    df["predicted_taxonomy_prob"] = np.nan
    df.loc[predict_mask, "predicted_taxonomy"]      = pred_labels
    df.loc[predict_mask, "predicted_taxonomy_prob"] = pred_proba

    print("  Predicted taxonomy distribution:")
    pvc = pd.Series(pred_labels).value_counts()
    for cls, n in pvc.items():
        print(f"    {cls:>8s}: {n:,}")

    # ── Save catalog ──────────────────────────────────────────────────────────
    # Drop the merged orbital columns that don't belong in the original catalog
    # (they are already in mpcorb_orbital_class.parquet)
    # But keep them if they didn't exist before
    orig_cols = list(pd.read_parquet(CAT_PATH, columns=["number_mp"]).columns)
    out_df = df.copy()
    out_df.to_parquet(OUT_CAT, index=False)
    print(f"\n  Saved: {OUT_CAT} ({len(out_df):,} rows)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    classes_ordered = ["C", "Other", "S", "X"]

    # 1. Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(classes_ordered)))
    ax.set_yticks(range(len(classes_ordered)))
    ax.set_xticklabels(classes_ordered, fontsize=11)
    ax.set_yticklabels(classes_ordered, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True (GASP)", fontsize=12)
    ax.set_title(f"5-fold CV Confusion Matrix  (acc={acc:.3f})", fontsize=12)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10)
    plt.tight_layout()
    p1 = PLOT_DIR / "19_confusion_matrix.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p1}")

    # 2. Feature importance bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.viridis_r(np.linspace(0.2, 0.85, len(importances_sorted)))
    bars = ax.barh(importances_sorted.index[::-1],
                   importances_sorted.values[::-1],
                   color=colors[::-1])
    ax.set_xlabel("Mean Decrease in Impurity", fontsize=12)
    ax.set_title("Random Forest Feature Importances", fontsize=12)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xlim(0, importances_sorted.max() * 1.18)
    plt.tight_layout()
    p2 = PLOT_DIR / "19_feature_importance.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p2}")

    # 3. Predicted taxonomy distribution pie
    pred_valid = df["predicted_taxonomy"].dropna()
    pie_counts = pred_valid.value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    wedge_colors = {"S": "#e07b39", "C": "#4a90d9", "X": "#7b7b7b", "Other": "#9ccc65"}
    colors_pie = [wedge_colors.get(c, "#cccccc") for c in pie_counts.index]
    wedges, texts, autotexts = ax.pie(
        pie_counts.values,
        labels=pie_counts.index,
        colors=colors_pie,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.82,
        textprops={"fontsize": 12},
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax.set_title(f"Predicted Taxonomy Distribution\n(N={len(pred_valid):,} objects with full features)",
                 fontsize=12)
    plt.tight_layout()
    p3 = PLOT_DIR / "19_taxonomy_distribution.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p3}")

    # ── Log file ──────────────────────────────────────────────────────────────
    log_path = LOG_DIR / "19_classifier_stats.txt"
    with open(log_path, "w") as f:
        f.write("GAPC Step 19 — Taxonomy Random Forest Classifier\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Catalog: {CAT_PATH}\n")
        f.write(f"Training objects: {len(train_full):,}\n")
        f.write(f"5-fold CV accuracy: {acc:.4f}\n\n")
        f.write("Classification report (5-fold CV):\n")
        f.write(report + "\n")
        f.write("Feature importances:\n")
        for feat, imp in importances_sorted.items():
            f.write(f"  {feat:>15s}: {imp:.4f}\n")
        f.write("\nPredicted taxonomy distribution (full catalog):\n")
        for cls, n in pvc.items():
            f.write(f"  {cls:>8s}: {n:,}\n")
        f.write(f"\nModel saved to: {MODEL_OUT}\n")
        f.write(f"Catalog saved to: {OUT_CAT}\n")
    print(f"\n  Log: {log_path}")

    print("\nStep 19 complete.")


if __name__ == "__main__":
    main()
