# -*- coding: utf-8 -*-
"""
Loan approval prediction - clean pipeline (revised after peer review)
Train (80%) -> fit models | Validation (10%) -> ensemble weights | Test (10%) -> final, evaluated once

Added in this revision (reviewer requests):
  1. Five-fold stratified CV results are saved as per-fold, summary (mean +/- std).
  2. Feature importance: Random Forest impurity importance + permutation importance (Random Forest and Stacking).
  3. Credit-score check: (a) single-feature credit-score baseline, (b) every model re-run WITHOUT cibil_score,
     both on the 80/10/10 test split and with 5-fold CV.

Usage: python loan_pipeline_clean.py loan_approval_dataset.csv
Outputs (CSV + one PNG) are written to ./results/
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

SEED = 42
CSV = sys.argv[1] if len(sys.argv) > 1 else "loan_approval_dataset.csv"
FEATURES = ["income_annum", "loan_amount", "cibil_score", "loan_term",
            "self_employed", "no_of_dependents",
            "commercial_assets_value", "bank_asset_value"]
BASE = ["Logistic Regression", "Decision Tree", "Random Forest", "KNN", "Neural Network"]
ORDER = BASE + ["Majority Voting", "Weighted Voting", "Stacking"]
CREDIT = "cibil_score"
OUT = "results"
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- 1. data (X, y defined ONCE)
def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    X = df[FEATURES].copy()
    X["self_employed"] = X["self_employed"].astype(str).str.strip().str.lower().map({"no": 0, "yes": 1})
    y = df["loan_status"].astype(str).str.strip().str.lower().map({"approved": 1, "rejected": 0})
    assert X.notna().all().all() and y.notna().all(), "unmapped values found"
    return X, y


# ---------------------------------------------------------------- 2. models
def make_base_models():
    return {
        "Logistic Regression": Pipeline([("s", StandardScaler()),
                                         ("m", LogisticRegression(max_iter=1000, random_state=SEED))]),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED),
        "KNN": Pipeline([("s", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=5))]),
        "Neural Network": Pipeline([("s", StandardScaler()),
                                    ("m", MLPClassifier(hidden_layer_sizes=(100, 50), activation="relu",
                                                        solver="adam", max_iter=1000, random_state=SEED))]),
    }


def make_stacking():
    est = [(n.lower().replace(" ", "_"), m) for n, m in make_base_models().items()]
    return StackingClassifier(estimators=est,
                              final_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
                              cv=5, stack_method="predict_proba", n_jobs=-1)


def metrics(y_true, y_pred):
    return {"Accuracy": accuracy_score(y_true, y_pred), "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred), "F1-Score": f1_score(y_true, y_pred)}


def majority(pred_matrix):
    return (pred_matrix.sum(axis=1) >= 3).astype(int)


def weighted(pred_matrix, w):
    return ((pred_matrix * w).sum(axis=1) >= 0.5).astype(int)


def fit_and_predict(X_tr, y_tr, X_val, y_val, eval_sets):
    """Fit on X_tr, derive weights from X_val ONLY, return predictions for every set in eval_sets."""
    models = make_base_models()
    for m in models.values():
        m.fit(X_tr, y_tr)
    # weights from VALIDATION precision
    val_prec = np.array([precision_score(y_val, models[n].predict(X_val)) for n in BASE])
    w = val_prec / val_prec.sum()
    stack = make_stacking().fit(X_tr, y_tr)
    out = {}
    for name, Xs in eval_sets.items():
        P = np.column_stack([models[n].predict(Xs) for n in BASE])
        d = {n: P[:, i] for i, n in enumerate(BASE)}
        d["Majority Voting"] = majority(P)
        d["Weighted Voting"] = weighted(P, w)
        d["Stacking"] = stack.predict(Xs)
        out[name] = d
    return out, w, models, stack


def run_cv(X, y, tag):
    """5-fold stratified CV of the whole pipeline (10% of each training fold used as validation for weights)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_rows = []
    for k, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        Xa, ya = X.iloc[tr_idx], y.iloc[tr_idx]
        Xf, yf = X.iloc[te_idx], y.iloc[te_idx]
        Xa_tr, Xa_val, ya_tr, ya_val = train_test_split(Xa, ya, test_size=1/9, random_state=SEED, stratify=ya)
        p, _, _, _ = fit_and_predict(Xa_tr, ya_tr, Xa_val, ya_val, {"fold": Xf})
        for model, pr in p["fold"].items():
            fold_rows.append({"Fold": k, "Model": model, **metrics(yf, pr)})
        print(f"  [{tag}] fold {k} done")
    cv = pd.DataFrame(fold_rows)
    cv.to_csv(f"{OUT}/cv_folds_{tag}.csv", index=False)
    cols = ["Accuracy", "Precision", "Recall", "F1-Score"]
    g = cv.groupby("Model")[cols].agg(["mean", "std"])
    summ = pd.DataFrame({m: [f"{g[(m, 'mean')][mod]:.4f} ± {g[(m, 'std')][mod]:.4f}" for mod in ORDER]
                         for m in cols}, index=ORDER)
    summ.index.name = "Model"
    summ.to_csv(f"{OUT}/cv_summary_{tag}.csv")
    return cv, summ


if __name__ == "__main__":
    X, y = load_data(CSV)
    print(f"Samples: {len(X)} | Approved: {(y==1).sum()} | Rejected: {(y==0).sum()}")

    # ------------------------------------------------------------ strict 80/10/10 split
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)
    print(f"Train {len(X_tr)} | Validation {len(X_val)} | Test {len(X_te)}")

    preds, w, models, stack = fit_and_predict(X_tr, y_tr, X_val, y_val,
                                              {"Training": X_tr, "Validation": X_val, "Test": X_te})
    labels = {"Training": y_tr, "Validation": y_val, "Test": y_te}

    print("\nWeighted-voting weights (from VALIDATION precision):")
    for n, wi in zip(BASE, w):
        print(f"  {n:20s} {wi:.4f}")
    pd.DataFrame({"Model": BASE, "Weight": w.round(4)}).to_csv(f"{OUT}/weighted_voting_weights.csv", index=False)

    rows = []
    for split in ["Training", "Validation", "Test"]:
        for model, p in preds[split].items():
            rows.append({"Split": split, "Model": model, **metrics(labels[split], p)})
    res = pd.DataFrame(rows).round(4)
    res.to_csv(f"{OUT}/results_split.csv", index=False)
    for split in ["Training", "Validation", "Test"]:
        print(f"\n===== {split.upper()} =====")
        print(res[res.Split == split].drop(columns="Split").to_string(index=False))

    print("\n===== TEST CONFUSION MATRICES (TN FP FN TP) =====")
    cm_rows = []
    for model, p in preds["Test"].items():
        tn, fp, fn, tp = confusion_matrix(y_te, p).ravel()
        cm_rows.append({"Model": model, "TN": tn, "FP": fp, "FN": fn, "TP": tp, "Errors": fp + fn})
    cm = pd.DataFrame(cm_rows)
    cm.to_csv(f"{OUT}/results_confusion.csv", index=False)
    print(cm.to_string(index=False))

    # ------------------------------------------------------------ REVIEW 1: 5-fold CV results
    print("\n===== 5-FOLD STRATIFIED CV, ALL FEATURES (mean ± std) =====")
    cv_full, summ_full = run_cv(X, y, "all_features")
    print(summ_full.to_string())

    # ------------------------------------------------------------ REVIEW 3a: feature importance
    print("\n===== FEATURE IMPORTANCE =====")
    rf = models["Random Forest"]
    imp = pd.DataFrame({"Feature": FEATURES, "RF impurity importance": rf.feature_importances_})
    for name, mdl in [("Random Forest", rf), ("Stacking", stack)]:
        pi = permutation_importance(mdl, X_te, y_te, scoring="accuracy", n_repeats=30,
                                    random_state=SEED, n_jobs=-1)
        imp[f"{name} permutation (accuracy drop)"] = pi.importances_mean
        imp[f"{name} permutation std"] = pi.importances_std
    imp = imp.sort_values("RF impurity importance", ascending=False).round(4)
    imp.to_csv(f"{OUT}/feature_importance.csv", index=False)
    print(imp.to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        s = imp.sort_values("RF impurity importance")
        ax[0].barh(s["Feature"], s["RF impurity importance"], color="#4C78A8")
        ax[0].set_title("Random Forest impurity importance")
        s = imp.sort_values("Stacking permutation (accuracy drop)")
        ax[1].barh(s["Feature"], s["Stacking permutation (accuracy drop)"],
                   xerr=s["Stacking permutation std"], color="#F58518")
        ax[1].set_title("Stacking permutation importance\n(test accuracy drop)")
        plt.tight_layout()
        plt.savefig(f"{OUT}/feature_importance.png", dpi=200)
        plt.close()
    except ImportError:
        print("matplotlib not installed - skipping the figure")

    # ------------------------------------------------------------ REVIEW 3b: credit-score checks
    print("\n===== CREDIT-SCORE CHECK =====")
    # (a) one-feature baseline
    base_rows = []
    for depth, name in [(1, "Credit score only - threshold (Decision Tree, depth 1)"),
                        (None, "Credit score only - Decision Tree (unrestricted)")]:
        m = DecisionTreeClassifier(max_depth=depth, random_state=SEED).fit(X_tr[[CREDIT]], y_tr)
        base_rows.append({"Model": name, **metrics(y_te, m.predict(X_te[[CREDIT]]))})
        if depth == 1:
            print(f"  depth-1 tree threshold: cibil_score <= {m.tree_.threshold[0]:.1f}")
    base = pd.DataFrame(base_rows).round(4)
    base.to_csv(f"{OUT}/credit_score_only_baseline.csv", index=False)
    print(base.to_string(index=False))

    # (b) every model re-run without credit score (same split, then CV)
    X_nc = X.drop(columns=[CREDIT])
    Xtr_nc, Xval_nc, Xte_nc = X_tr.drop(columns=[CREDIT]), X_val.drop(columns=[CREDIT]), X_te.drop(columns=[CREDIT])
    preds_nc, _, _, _ = fit_and_predict(Xtr_nc, y_tr, Xval_nc, y_val, {"Test": Xte_nc})
    nc = pd.DataFrame([{"Model": m, **metrics(y_te, p)} for m, p in preds_nc["Test"].items()]).round(4)
    nc.to_csv(f"{OUT}/test_without_credit_score.csv", index=False)
    print("\n  TEST SET, credit score REMOVED:")
    print(nc.to_string(index=False))

    print("\n===== 5-FOLD STRATIFIED CV, credit score REMOVED (mean ± std) =====")
    cv_nc, summ_nc = run_cv(X_nc, y, "without_credit_score")
    print(summ_nc.to_string())

    print(f"\nAll result files saved in ./{OUT}/")
