# ============================================================
# train_model.py
# Local training script for Matematik Pintar SPM
# Final model: Profile-Aware Hybrid RF-DNN
# ============================================================

from pathlib import Path
from collections import Counter
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_PATH = Path("data") / "student_performance.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

LEVEL_ENCODING = {"Rendah": 0, "Sederhana": 1, "Tinggi": 2}
LEVEL_DECODING = {0: "Rendah", 1: "Sederhana", 2: "Tinggi"}

# Raw FinalGrade in this dataset is reverse-coded:
# FinalGrade 0 has the strongest score range, while FinalGrade 3 has the weakest.
FINALGRADE_TO_LEVEL = {
    0: "Tinggi",
    1: "Sederhana",
    2: "Sederhana",
    3: "Rendah",
}

# ExamScore is kept in the CSV for audit/context, but excluded from final training.
# It is too strongly tied to FinalGrade and can cause unrealistic 100% accuracy.
INCLUDE_EXAM_SCORE = False

NUMERIC_FEATURES = [
    "StudyHours", "Attendance", "Age", "OnlineCourses", "AssignmentCompletion"
]

if INCLUDE_EXAM_SCORE:
    NUMERIC_FEATURES = NUMERIC_FEATURES + ["ExamScore"]

CATEGORICAL_FEATURES = [
    "Resources", "Extracurricular", "Motivation", "Internet", "Gender",
    "LearningStyle", "Discussions", "EduTech", "StressLevel"
]

FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = sorted(set(FEATURE_COLS + ["ExamScore", "FinalGrade"]))


def to_dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def evaluate_model(model_name, y_true, y_pred, dataset_name):
    return {
        "Dataset": dataset_name,
        "Model": model_name,
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def main():
    print("=== TRAINING PROFILE-AWARE HYBRID RF-DNN MODEL ===")
    print("Loading dataset:", DATA_PATH)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print("Loaded student dataset:", df.shape)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError("Missing columns in dataset: " + ", ".join(missing_cols))

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed duplicate rows: {before - len(df)} | Remaining: {len(df)}")

    # Keep only rows with valid target and required features.
    df = df.dropna(subset=REQUIRED_COLUMNS).copy()
    df["CompetencyText"] = df["FinalGrade"].astype(int).map(FINALGRADE_TO_LEVEL)
    df["CompetencyLevel"] = df["CompetencyText"].map(LEVEL_ENCODING).astype(int)

    # Audit print so it is clear ExamScore is not used as the final feature.
    score_grade_table = df.groupby("FinalGrade")["ExamScore"].agg(["min", "max", "mean", "count"]).reset_index()
    print("\nExamScore audit by raw FinalGrade:")
    print(score_grade_table.to_string(index=False))
    corr_value = df["ExamScore"].corr(df["FinalGrade"])
    print("Correlation between ExamScore and raw FinalGrade:", round(float(corr_value), 4))
    print("INCLUDE_EXAM_SCORE:", INCLUDE_EXAM_SCORE)

    X = df[FEATURE_COLS].copy()
    y = df["CompetencyLevel"].astype(int)

    print("\nFeature columns:", FEATURE_COLS)
    print("Target distribution:", dict(Counter(y)))

    # 80/10/10 stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    print("\nSplit sizes:")
    print("Train:", X_train.shape, "Validation:", X_val.shape, "Test:", X_test.shape)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", make_one_hot_encoder(), CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    print("Processed training shape:", X_train_processed.shape)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)
    print("\nSMOTE applied on training data only.")
    print("Before SMOTE:", dict(Counter(y_train)))
    print("After SMOTE :", dict(Counter(y_train_balanced)))

    # Random Forest tuning
    print("\n=== RANDOM FOREST TUNING ===")
    rf_base = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", bootstrap=True, n_jobs=-1)
    rf_param_grid = {
        "n_estimators": [200, 300, 500],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }
    rf_search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=rf_param_grid,
        n_iter=20,
        scoring="f1_weighted",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    rf_search.fit(X_train_balanced, y_train_balanced)
    rf_model = rf_search.best_estimator_
    print("Best RF parameters:", rf_search.best_params_)
    print("Best RF CV weighted F1:", round(float(rf_search.best_score_), 4))

    # Direct DNN / MLP baseline
    print("\n=== DIRECT DNN / MLP BASELINE ===")
    direct_dnn_model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=0.0005,
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
    )
    direct_dnn_model.fit(to_dense(X_train_balanced), y_train_balanced)
    print("Direct DNN/MLP trained. Iterations:", direct_dnn_model.n_iter_)

    # Hybrid RF-DNN input creation
    print("\n=== PROFILE-AWARE HYBRID RF-DNN TUNING ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rf_oof_probs = cross_val_predict(
        rf_model,
        X_train_balanced,
        y_train_balanced,
        cv=skf,
        method="predict_proba",
        n_jobs=-1,
    )
    rf_val_probs = rf_model.predict_proba(X_val_processed)
    rf_test_probs = rf_model.predict_proba(X_test_processed)

    X_train_hybrid = np.hstack([to_dense(X_train_balanced), rf_oof_probs])
    X_val_hybrid = np.hstack([to_dense(X_val_processed), rf_val_probs])
    X_test_hybrid = np.hstack([to_dense(X_test_processed), rf_test_probs])

    print(
        f"Hybrid input dimension: {X_train_hybrid.shape[1]} "
        f"({to_dense(X_train_balanced).shape[1]} profile features + 3 RF probabilities)"
    )

    search_configs = [
        {"arch": (64, 32), "alpha": 1e-4, "lr": 1e-3, "batch": 64},
        {"arch": (128, 64), "alpha": 1e-4, "lr": 1e-3, "batch": 64},
        {"arch": (128, 64, 32), "alpha": 1e-4, "lr": 1e-3, "batch": 64},
        {"arch": (256, 128), "alpha": 1e-4, "lr": 1e-3, "batch": 64},
        {"arch": (256, 128, 64), "alpha": 1e-4, "lr": 1e-3, "batch": 64},
        {"arch": (384, 192, 96), "alpha": 1e-4, "lr": 1e-3, "batch": 64},
        {"arch": (512, 256, 128), "alpha": 1e-4, "lr": 5e-4, "batch": 64},
        {"arch": (128, 64, 32), "alpha": 5e-4, "lr": 1e-3, "batch": 64},
        {"arch": (256, 128, 64), "alpha": 5e-4, "lr": 1e-3, "batch": 64},
        {"arch": (384, 192, 96), "alpha": 5e-4, "lr": 5e-4, "batch": 64},
        {"arch": (128, 64), "alpha": 1e-5, "lr": 1e-3, "batch": 64},
        {"arch": (128, 64, 32), "alpha": 1e-5, "lr": 1e-3, "batch": 64},
        {"arch": (256, 128, 64), "alpha": 1e-5, "lr": 5e-4, "batch": 64},
        {"arch": (128, 64, 32), "alpha": 1e-4, "lr": 5e-4, "batch": 128},
        {"arch": (256, 128, 64), "alpha": 1e-4, "lr": 5e-4, "batch": 128},
        {"arch": (384, 192, 96), "alpha": 1e-4, "lr": 3e-4, "batch": 128},
    ]

    best_hybrid_model = None
    best_architecture = None
    best_hybrid_config = None
    best_val_f1 = -1
    hybrid_search_records = []

    print("Hybrid DNN search based on validation F1:")
    for cfg in search_configs:
        model = MLPClassifier(
            hidden_layer_sizes=cfg["arch"],
            activation="relu",
            solver="adam",
            alpha=cfg["alpha"],
            learning_rate_init=cfg["lr"],
            batch_size=cfg["batch"],
            max_iter=800,
            early_stopping=True,
            n_iter_no_change=25,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train_hybrid, y_train_balanced)
        val_pred = model.predict(X_val_hybrid)
        val_f1 = f1_score(y_val, val_pred, average="weighted", zero_division=0)
        hybrid_search_records.append({
            "Architecture": str(cfg["arch"]),
            "Alpha": cfg["alpha"],
            "Learning Rate": cfg["lr"],
            "Batch Size": cfg["batch"],
            "Validation F1": float(val_f1),
            "Iterations Run": int(model.n_iter_),
            "Final Loss": float(getattr(model, "loss_", getattr(model, "loss", 0))),
        })
        print(
            f"arch={str(cfg['arch']):18s} alpha={cfg['alpha']} "
            f"lr={cfg['lr']} batch={cfg['batch']} -> val F1={val_f1:.4f}"
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_hybrid_model = model
            best_architecture = cfg["arch"]
            best_hybrid_config = cfg

    hybrid_model = best_hybrid_model
    hybrid_search_df = pd.DataFrame(hybrid_search_records).sort_values("Validation F1", ascending=False)
    hybrid_search_df.to_csv(MODEL_DIR / "hybrid_tuning_results.csv", index=False)

    print("Selected architecture:", best_architecture)
    print("Selected alpha:", best_hybrid_config["alpha"])
    print("Selected learning rate:", best_hybrid_config["lr"])
    print("Selected batch size:", best_hybrid_config["batch"])
    print("Best validation F1:", round(float(best_val_f1), 4))

    # Evaluation
    rf_val_pred = rf_model.predict(X_val_processed)
    rf_test_pred = rf_model.predict(X_test_processed)
    dnn_val_pred = direct_dnn_model.predict(to_dense(X_val_processed))
    dnn_test_pred = direct_dnn_model.predict(to_dense(X_test_processed))
    hybrid_val_pred = hybrid_model.predict(X_val_hybrid)
    hybrid_test_pred = hybrid_model.predict(X_test_hybrid)

    results = []
    for dataset_name, y_true, preds in [
        ("Validation", y_val, {
            "Random Forest": rf_val_pred,
            "Direct DNN/MLP": dnn_val_pred,
            "Profile-Aware Hybrid RF-DNN": hybrid_val_pred,
        }),
        ("Test", y_test, {
            "Random Forest": rf_test_pred,
            "Direct DNN/MLP": dnn_test_pred,
            "Profile-Aware Hybrid RF-DNN": hybrid_test_pred,
        }),
    ]:
        for model_name, pred in preds.items():
            results.append(evaluate_model(model_name, y_true, pred, dataset_name))

    results_df = pd.DataFrame(results)
    print("\n=== MODEL COMPARISON ===")
    print(results_df.to_string(index=False))
    results_df.to_csv(MODEL_DIR / "model_comparison_results.csv", index=False)

    print("\n=== FINAL HYBRID TEST REPORT ===")
    print(classification_report(y_test, hybrid_test_pred, target_names=["Rendah", "Sederhana", "Tinggi"], zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, hybrid_test_pred))

    # Feature importance from RF component
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(to_dense(X_train_balanced).shape[1])]
    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf_model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    feature_importance_df.to_csv(MODEL_DIR / "feature_importance.csv", index=False)

    summary = {
        "final_model": "Profile-Aware Hybrid RF-DNN",
        "include_exam_score": INCLUDE_EXAM_SCORE,
        "feature_cols": FEATURE_COLS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "level_encoding": LEVEL_ENCODING,
        "level_decoding": LEVEL_DECODING,
        "best_rf_params": rf_search.best_params_,
        "best_hybrid_architecture": str(best_architecture),
        "best_hybrid_config": best_hybrid_config,
        "best_hybrid_validation_f1": float(best_val_f1),
        "results": results,
        "feature_note": "ExamScore is excluded from final model training after checking its strong relationship with FinalGrade.",
    }
    with open(MODEL_DIR / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    bundle = {
        "model_type": "Profile-Aware Hybrid RF-DNN",
        "preprocessor": preprocessor,
        "rf_model": rf_model,
        "direct_dnn_model": direct_dnn_model,
        "hybrid_model": hybrid_model,
        "feature_cols": FEATURE_COLS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "level_encoding": LEVEL_ENCODING,
        "level_decoding": LEVEL_DECODING,
        "include_exam_score": INCLUDE_EXAM_SCORE,
        "smote_used": True,
        "best_rf_params": rf_search.best_params_,
        "best_hybrid_architecture": best_architecture,
        "best_hybrid_config": best_hybrid_config,
    }
    joblib.dump(bundle, MODEL_DIR / "hybrid_rf_dnn_bundle.pkl")

    print("\nSaved model files:")
    print("- models/hybrid_rf_dnn_bundle.pkl")
    print("- models/metrics_summary.json")
    print("- models/model_comparison_results.csv")
    print("- models/feature_importance.csv")
    print("- models/hybrid_tuning_results.csv")


if __name__ == "__main__":
    main()
