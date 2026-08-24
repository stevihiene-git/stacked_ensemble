"""
Reddit Suspicious URL Detection: Stacked Ensemble (Super Learner) Pipeline
============================================================================
Implements the methodology described in Chapter 3 of the thesis:
  - 6 baseline (first-level) classifiers: AdaBoost, Gradient Boost, Random
    Forest, Linear SVM, Decision Tree, Naive Bayes
  - A stacked ensemble (Super Learner) built via 5-fold out-of-fold
    predictions, with 4 meta-learner variants:
      1. Logistic Regression meta-learner
      2. Majority Voting (uniform weights)
      3. Weighted Voting (accuracy-based weights)
      4. Weighted Voting (exponential accuracy weights, lambda=10)
  - Evaluation metrics: Accuracy, Precision, Recall, F-score, FPR, AUC, MCC
  - Produces Table 4.1, Table 4.2, Table 4.3 (as CSV) and an ROC comparison
    chart, ready to drop into Chapter 4.

USAGE
-----
    python reddit_phishing_pipeline.py --data path/to/reddit_data.csv --out results/

If --data is omitted, or the file doesn't exist, a small synthetic dataset
matching the expected schema is generated automatically so you can smoke-test
the whole pipeline before your real Reddit dataset is ready.

EXPECTED CSV SCHEMA (12 features from Table 3.1 + label column)
-----------------------------------------------------------------
username_length, username_has_digits, username_digit_count, account_age_days,
post_content_signal, domain_age, is_https, link_length, link_char_count,
link_dot_count, link_special_char_count, link_digit_count, label

`label` must be 0 (legitimate) or 1 (malicious/phishing).
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler as MMS
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, matthews_corrcoef,
)

warnings.filterwarnings("ignore")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# CORRECTED feature schema. The original reddit_post_features.csv export
# computed all URL-lexical features (LinkLength, IsSecured, NumDots, etc.)
# on the Reddit permalink column, which is IDENTICAL for every row
# structurally (always reddit.com/r/<sub>/comments/<id>/...) rather than on
# the actual outbound URL the user posted -- this was verified directly
# (100% of URL column values contained 'reddit.com') and explains why those
# features showed ~0 correlation with the phishing label. This corrected
# pipeline instead extracts the real embedded external URL from each post's
# raw Body/Title text and computes lexical features on THAT url. 'domain_age'
# (WHOIS-based) remains unavailable, as before.
RAW_FEATURE_COLUMNS = [
    "link_length", "is_https", "num_dots", "num_digits", "num_special_chars", "domain_length",
    "username_length", "username_digit_count", "title_length", "post_score", "num_comments",
    "account_age", "subreddit_phish_rate",
]
RAW_LABEL_COLUMN = "label"  # already 0/1 in the corrected, pre-built dataset

FEATURE_COLUMNS = RAW_FEATURE_COLUMNS  # used throughout the rest of the pipeline
LABEL_COLUMN = "label"

CLASS_BALANCE_RATIO = 4  # legitimate : malicious, matching Azeez et al. (2022) and Chapter 3.3


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    """Loads the corrected Reddit dataset (real extracted-URL features +
    Reddit-side behavioural features), and applies the 4:1
    legitimate:malicious class-balancing under-sampling described in
    Chapter 3.3. Falls back to a synthetic dataset if csv_path is missing,
    purely for smoke-testing."""
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path, encoding="latin1", on_bad_lines="warn")
        missing = set(RAW_FEATURE_COLUMNS + [RAW_LABEL_COLUMN]) - set(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")

        df = df.drop_duplicates(subset=["PostID"]).copy()
        df[LABEL_COLUMN] = df[RAW_LABEL_COLUMN].astype(int)

        # Apply 4:1 class balancing (under-sample the legitimate/majority class)
        n_malicious = (df[LABEL_COLUMN] == 1).sum()
        n_legit_target = min((df[LABEL_COLUMN] == 0).sum(), n_malicious * CLASS_BALANCE_RATIO)
        malicious_df = df[df[LABEL_COLUMN] == 1]
        legit_df = df[df[LABEL_COLUMN] == 0].sample(n=n_legit_target, random_state=RANDOM_SEED)
        df_balanced = pd.concat([malicious_df, legit_df], ignore_index=True)
        df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

        print(f"Loaded corrected real dataset: {df.shape[0]} unique rows from {csv_path}")
        print(f"Applied 4:1 class balancing: {n_legit_target} legitimate + {n_malicious} malicious "
              f"= {df_balanced.shape[0]} total rows")
        return df_balanced

    print("No dataset found at the given path -- generating a SYNTHETIC "
          "dataset instead, purely to smoke-test the pipeline. "
          "Replace --data with your real Reddit CSV before reporting results.")
    return _generate_synthetic_dataset()


def _generate_synthetic_dataset(n_legit=4000, n_phish=1000) -> pd.DataFrame:
    """Generates a synthetic dataset with the 4:1 class ratio used in
    Azeez et al. (2022), with phishing samples drawn from shifted
    distributions so the classifiers have genuine signal to learn from."""
    rng = np.random.default_rng(RANDOM_SEED)

    def make_class(n, malicious: bool):
        shift = 1.0 if malicious else 0.0
        return pd.DataFrame({
            "username_length": rng.normal(10 + 3 * shift, 3, n).clip(3, 30),
            "username_has_digits": rng.binomial(1, 0.3 + 0.3 * shift, n),
            "username_digit_count": rng.poisson(1 + 2 * shift, n),
            "account_age_days": rng.exponential(600 - 400 * shift, n).clip(0, 4000),
            "post_content_signal": rng.normal(0.3 + 0.4 * shift, 0.2, n).clip(0, 1),
            "domain_age": rng.exponential(900 - 600 * shift, n).clip(0, 5000),
            "is_https": rng.binomial(1, 0.85 - 0.35 * shift, n),
            "link_length": rng.normal(25 + 15 * shift, 8, n).clip(8, 120),
            "link_char_count": rng.normal(20 + 12 * shift, 6, n).clip(5, 100),
            "link_dot_count": rng.poisson(2 + 2 * shift, n),
            "link_special_char_count": rng.poisson(1 + 3 * shift, n),
            "link_digit_count": rng.poisson(2 + 3 * shift, n),
            "label": int(malicious),
        })

    df = pd.concat([make_class(n_legit, False), make_class(n_phish, True)], ignore_index=True)
    return df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].values
    y = df[LABEL_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


# ---------------------------------------------------------------------------
# 3. Baseline (first-level) classifiers
# ---------------------------------------------------------------------------
def get_base_learners():
    """Returns the 6 base learners with small hyperparameter grids for
    GridSearchCV, matching Table 3.2. LinearSVC is wrapped in
    CalibratedClassifierCV so it can output predict_proba, which the
    stacking procedure requires."""
    return {
        "AdaBoost": (
            AdaBoostClassifier(random_state=RANDOM_SEED),
            {"n_estimators": [50, 100], "learning_rate": [0.5, 1.0]},
        ),
        "Gradient Boost": (
            GradientBoostingClassifier(random_state=RANDOM_SEED),
            {"n_estimators": [100], "learning_rate": [0.1], "max_depth": [3]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_SEED),
            {"n_estimators": [100, 200], "max_depth": [None, 10]},
        ),
        "Linear SVM": (
            CalibratedClassifierCV(LinearSVC(random_state=RANDOM_SEED, max_iter=5000), cv=3),
            {"estimator__C": [0.1, 1.0]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_SEED),
            {"criterion": ["entropy"], "max_depth": [None, 10]},
        ),
        "Naive Bayes": (
            GaussianNB(),
            {"var_smoothing": [1e-9, 1e-8]},
        ),
    }


def train_baseline_models(X_train, y_train):
    """Tunes and fits each base learner with 5-fold CV, optimizing F1."""
    fitted = {}
    for name, (estimator, grid) in get_base_learners().items():
        print(f"Training base learner: {name} ...")
        gs = GridSearchCV(estimator, grid, cv=5, scoring="f1", n_jobs=-1)
        gs.fit(X_train, y_train)
        fitted[name] = gs.best_estimator_
        print(f"  best params: {gs.best_params_}")
    return fitted


# ---------------------------------------------------------------------------
# 4. Stacked ensemble (Super Learner)
# ---------------------------------------------------------------------------
def build_meta_features(models: dict, X, y=None, use_cv=False):
    """Builds the meta-feature matrix Z (probability of the positive class
    from each base learner). If use_cv=True, uses 5-fold out-of-fold
    predictions on the training set (required to avoid leakage); otherwise
    uses each model's direct predict_proba (for the test set)."""
    cols = []
    for name, model in models.items():
        if use_cv:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
            probs = cross_val_predict(model, X, y, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]
        else:
            probs = model.predict_proba(X)[:, 1]
        cols.append(probs)
    return np.column_stack(cols)


def train_meta_learners(Z_train, y_train, base_train_accuracy: dict):
    """Trains the 4 meta-learner variants described in Section 3.6."""
    meta_learners = {}

    # 1. Logistic Regression meta-learner
    lr = LogisticRegression(random_state=RANDOM_SEED)
    lr.fit(Z_train, y_train)
    meta_learners["Logistic Regression (meta)"] = ("model", lr)

    # 2. Majority Voting (uniform weights)
    n_learners = Z_train.shape[1]
    uniform_weights = np.ones(n_learners) / n_learners
    meta_learners["Majority Voting (uniform)"] = ("weights", uniform_weights)

    # 3. Weighted Voting (accuracy-based weights)
    accs = np.array(list(base_train_accuracy.values()))
    acc_weights = accs / accs.sum()
    meta_learners["Weighted Voting (accuracy)"] = ("weights", acc_weights)

    # 4. Weighted Voting (exponential accuracy weights, lambda=10)
    lam = 10
    exp_weights = np.exp(lam * accs)
    exp_weights = exp_weights / exp_weights.sum()
    meta_learners["Weighted Voting (exponential)"] = ("weights", exp_weights)

    return meta_learners


def predict_with_meta_learner(meta_learner, Z):
    kind, obj = meta_learner
    if kind == "model":
        return obj.predict_proba(Z)[:, 1]
    else:  # weighted combination
        return Z @ obj


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------
def evaluate(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F-score": f1_score(y_true, y_pred, zero_division=0),
        "FPR": fpr,
        "AUC": roc_auc_score(y_true, y_prob),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# 6. Main pipeline
# ---------------------------------------------------------------------------
def main(data_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # --- Load and preprocess ---
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess(df)
    print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")
    print(f"Train class balance: {np.bincount(y_train)}  |  Test class balance: {np.bincount(y_test)}\n")

    # --- Train baseline (first-level) classifiers ---
    base_models = train_baseline_models(X_train, y_train)

    # Table 4.1: standalone classifier results on the held-out test set
    baseline_results = {}
    base_train_accuracy = {}
    for name, model in base_models.items():
        test_probs = model.predict_proba(X_test)[:, 1]
        baseline_results[name] = evaluate(y_test, test_probs)
        base_train_accuracy[name] = accuracy_score(y_train, model.predict(X_train))

    table_4_1 = pd.DataFrame(baseline_results).T
    table_4_1.to_csv(os.path.join(out_dir, "table_4_1_baseline_results.csv"))
    print("=== Table 4.1: Baseline Classifier Results ===")
    print(table_4_1.round(4), "\n")

    # --- Build stacked ensemble ---
    print("Building out-of-fold meta-features for stacking (this may take a while)...")
    Z_train = build_meta_features(base_models, X_train, y_train, use_cv=True)
    Z_test = build_meta_features(base_models, X_test, use_cv=False)

    meta_learners = train_meta_learners(Z_train, y_train, base_train_accuracy)

    # Table 4.2: stacked ensemble variant results
    ensemble_results = {}
    ensemble_probs = {}
    for name, ml in meta_learners.items():
        probs = predict_with_meta_learner(ml, Z_test)
        ensemble_probs[name] = probs
        ensemble_results[name] = evaluate(y_test, probs)

    table_4_2 = pd.DataFrame(ensemble_results).T
    table_4_2.to_csv(os.path.join(out_dir, "table_4_2_stacked_ensemble_results.csv"))
    print("=== Table 4.2: Stacked Ensemble Variant Results ===")
    print(table_4_2.round(4), "\n")

    # --- Table 4.3: best standalone vs. best stacked ensemble ---
    best_baseline_name = table_4_1["F-score"].idxmax()
    best_ensemble_name = table_4_2["F-score"].idxmax()
    delta = table_4_2.loc[best_ensemble_name] - table_4_1.loc[best_baseline_name]

    table_4_3 = pd.DataFrame({
        f"Best standalone: {best_baseline_name}": table_4_1.loc[best_baseline_name],
        f"Best ensemble: {best_ensemble_name}": table_4_2.loc[best_ensemble_name],
        "Improvement (delta)": delta,
    }).T
    table_4_3.to_csv(os.path.join(out_dir, "table_4_3_comparison.csv"))
    print("=== Table 4.3: Best Standalone vs. Best Stacked Ensemble ===")
    print(table_4_3.round(4), "\n")

    # --- ROC curve comparison chart ---
    plt.figure(figsize=(8, 6))
    for name, model in base_models.items():
        probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, linestyle="--", alpha=0.6, label=f"{name} (base)")

    best_probs = ensemble_probs[best_ensemble_name]
    fpr, tpr, _ = roc_curve(y_test, best_probs)
    plt.plot(fpr, tpr, linewidth=2.5, color="black", label=f"{best_ensemble_name} (best ensemble)")

    plt.plot([0, 1], [0, 1], linestyle=":", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison: Base Classifiers vs. Best Stacked Ensemble")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(out_dir, "figure_4_roc_comparison.png")
    plt.savefig(roc_path, dpi=300)
    print(f"ROC comparison chart saved to: {roc_path}")

    print(f"\nAll results saved to: {out_dir}/")
    print("Files: table_4_1_baseline_results.csv, table_4_2_stacked_ensemble_results.csv, "
          "table_4_3_comparison.csv, figure_4_roc_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reddit phishing URL stacked ensemble pipeline")
    parser.add_argument("--data", type=str, default="", help="Path to Reddit dataset CSV")
    parser.add_argument("--out", type=str, default="results", help="Output directory")
    args = parser.parse_args()
    main(args.data, args.out)
