# ============================================================
# train_model.py
# Local training script for Matematik Pintar SPM
# Final model: Hybrid RF-DNN (ExamScore banded + profile features)
# ============================================================
#
# One model, used everywhere: this script trains the single Hybrid RF-DNN that
# is evaluated in the report AND loaded by app.py. There is no separate
# "deployment" model.
#
# ExamScore is audited below and found to have a ~0.97 correlation with the raw
# label -- using it as a precise 0-100 number would make the model a trivial
# lookup (100% accuracy, ~87% of feature importance on that one column, every
# behavioral feature <1%). Excluding it entirely, on the other hand, was tried
# first and found to produce an unstable model: SHAP and a feature-sensitivity
# sweep showed small, realistic changes in Attendance or AssignmentCompletion
# could flip a student's predicted level with high confidence in both
# directions, because none of the behavioral features have real signal on
# their own (all correlate under 0.04 with the label).
#
# The fix used here is to keep ExamScore, but bucket it into 4 coarse bands
# instead of a precise score. Two of the four bands straddle a label boundary,
# so the model still has genuine, honest signal from ExamScore while leaving
# real (if modest) room for the behavioral features to matter -- and heavier
# MLP regularization keeps the resulting decision surface smooth.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

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

# ExamScore never goes below 40 in this dataset (Rendah's true range is 40-54), so a band
# edge below 40 would have zero training examples -- a real user entering a low score would
# hit an unseen one-hot category and the model would fall back on other features, producing a
# nonsensical prediction. The lowest band's upper edge is therefore 40, not a smaller number.
EXAM_BAND_EDGES = [-1, 40, 60, 80, 100]
EXAM_BAND_LABELS = [0, 1, 2, 3]


def exam_score_to_band(score):
    score = max(0.0, min(100.0, float(score)))
    for i in range(len(EXAM_BAND_EDGES) - 1):
        if EXAM_BAND_EDGES[i] < score <= EXAM_BAND_EDGES[i + 1]:
            return EXAM_BAND_LABELS[i]
    return EXAM_BAND_LABELS[0]


NUMERIC_FEATURES = [
    "StudyHours", "Attendance", "Age", "OnlineCourses", "AssignmentCompletion"
]
CATEGORICAL_FEATURES = [
    "Resources", "Extracurricular", "Motivation", "Internet", "Gender",
    "LearningStyle", "Discussions", "EduTech", "StressLevel", "ExamBand"
]

FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = sorted(set(FEATURE_COLS + ["ExamScore", "FinalGrade"]) - {"ExamBand"})


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


def check_stability(preprocessor, rf_model, hybrid_model):
    """Sweep single features on a fixed baseline profile and print predictions, so a chaotic
    (non-monotonic, high-confidence-flip) decision surface would be caught here rather than
    discovered later in the running app."""
    base_profile = {
        "StudyHours": 18, "Attendance": 85, "Resources": 1, "Extracurricular": 1,
        "Motivation": 2, "Internet": 1, "Gender": 1, "Age": 18, "LearningStyle": 1,
        "OnlineCourses": 4, "Discussions": 1, "AssignmentCompletion": 80,
        "EduTech": 1, "StressLevel": 1, "ExamScore": 50,  # ambiguous band (Rendah/Sederhana)
    }

    def predict(profile):
        profile = dict(profile)
        profile["ExamBand"] = exam_score_to_band(profile.pop("ExamScore"))
        df_in = pd.DataFrame([profile])[FEATURE_COLS]
        X_proc = to_dense(preprocessor.transform(df_in))
        rf_probs = rf_model.predict_proba(X_proc)
        X_hybrid = np.hstack([X_proc, rf_probs])
        return hybrid_model.predict_proba(X_hybrid)[0]

    print("\n=== STABILITY CHECK (sweeping one feature at a time) ===")
    print("ExamScore sweep (intended primary driver -- should move smoothly, in bands):")
    for score in [10, 20, 30, 40, 50, 54, 55, 65, 75, 84, 85, 95]:
        p = dict(base_profile); p["ExamScore"] = score
        probs = predict(p)
        print(f"  ExamScore={score:3d} (band={exam_score_to_band(score)}) -> {LEVEL_DECODING[int(np.argmax(probs))]:10s} {np.round(probs, 3)}")

    print("Attendance sweep at a fixed ambiguous ExamScore=50 (should shift smoothly, no wild flips):")
    for att in [20, 40, 60, 80, 85, 100]:
        p = dict(base_profile); p["Attendance"] = att
        probs = predict(p)
        print(f"  Attendance={att:3d} -> {LEVEL_DECODING[int(np.argmax(probs))]:10s} {np.round(probs, 3)}")

    print("AssignmentCompletion sweep at a fixed ambiguous ExamScore=50 (should shift smoothly, no wild flips):")
    for ac in [50, 60, 70, 80, 90, 100]:
        p = dict(base_profile); p["AssignmentCompletion"] = ac
        probs = predict(p)
        print(f"  AssignmentCompletion={ac:3d} -> {LEVEL_DECODING[int(np.argmax(probs))]:10s} {np.round(probs, 3)}")


def run_shap_analysis(preprocessor, rf_model, hybrid_model, X_train_hybrid, X_test_hybrid,
                       hybrid_test_pred, level_decoding):
    """SHAP explanation of the final calibrated Hybrid RF-DNN: a handful of individual case
    studies (one per predicted class) plus a global mean-|SHAP| feature ranking over a sampled
    slice of the test set. This is the same technique used during the original instability
    investigation, run here on the FINAL (banded ExamScore) model to confirm its behaviour is
    sensible and to give a second, independent feature-importance view alongside the Random
    Forest's own Gini importance."""
    import shap

    print("\n=== SHAP ANALYSIS ===")
    feature_names = list(preprocessor.get_feature_names_out()) + [
        "RF_prob_Rendah", "RF_prob_Sederhana", "RF_prob_Tinggi"
    ]

    rng = np.random.RandomState(RANDOM_STATE)
    bg_idx = rng.choice(len(X_train_hybrid), size=min(300, len(X_train_hybrid)), replace=False)
    background = shap.kmeans(X_train_hybrid[bg_idx], 25)
    explainer = shap.KernelExplainer(hybrid_model.predict_proba, background)

    # Case studies: one real test-set profile per predicted class.
    case_records = []
    for cls in [0, 1, 2]:
        idx = np.where(hybrid_test_pred == cls)[0]
        if len(idx) == 0:
            continue
        row = idx[0]
        instance = X_test_hybrid[row:row + 1]
        sv = np.array(explainer.shap_values(instance, nsamples=150))  # (1, n_features, n_classes)
        contrib = pd.DataFrame({
            "feature": feature_names,
            "value": instance[0],
            "shap_value": sv[0, :, cls],
        }).sort_values("shap_value", key=np.abs, ascending=False)
        print(f"\nCase study -- predicted {level_decoding[cls]} (test row {row}):")
        print(contrib.head(8).to_string(index=False))
        case_records.append({
            "test_row": int(row),
            "predicted": level_decoding[cls],
            "top_features": contrib.head(8).to_dict(orient="records"),
        })

    with open(MODEL_DIR / "shap_case_studies.json", "w", encoding="utf-8") as f:
        json.dump(case_records, f, indent=2, default=float)

    # Global importance: mean |SHAP| over a sampled slice of the test set (kept small --
    # KernelExplainer is expensive -- but large enough for a stable ranking).
    global_idx = rng.choice(len(X_test_hybrid), size=min(40, len(X_test_hybrid)), replace=False)
    sv_global = np.array(explainer.shap_values(X_test_hybrid[global_idx], nsamples=100))
    mean_abs = np.abs(sv_global).mean(axis=(0, 2))
    global_importance = pd.DataFrame({
        "Feature": feature_names, "MeanAbsSHAP": mean_abs
    }).sort_values("MeanAbsSHAP", ascending=False)
    global_importance.to_csv(MODEL_DIR / "shap_global_importance.csv", index=False)
    print("\nTop 10 features by mean |SHAP| (global, sampled test set):")
    print(global_importance.head(10).to_string(index=False))

    return case_records, global_importance


def main():
    print("=== TRAINING HYBRID RF-DNN MODEL ===")
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

    df = df.dropna(subset=REQUIRED_COLUMNS).copy()
    df["CompetencyText"] = df["FinalGrade"].astype(int).map(FINALGRADE_TO_LEVEL)
    df["CompetencyLevel"] = df["CompetencyText"].map(LEVEL_ENCODING).astype(int)

    # ExamScore audit -- shows why the raw score can't be used directly, and sets up the
    # banding decision below.
    score_grade_table = df.groupby("FinalGrade")["ExamScore"].agg(["min", "max", "mean", "count"]).reset_index()
    print("\nExamScore audit by raw FinalGrade:")
    print(score_grade_table.to_string(index=False))
    corr_value = df["ExamScore"].corr(df["FinalGrade"])
    print("Correlation between ExamScore and raw FinalGrade:", round(float(corr_value), 4))

    df["ExamBand"] = df["ExamScore"].apply(exam_score_to_band)
    print("\nExamBand vs label crosstab (bands straddling a label boundary give the behavioral")
    print("features genuine room to matter, instead of ExamScore alone deciding):")
    print(pd.crosstab(df["ExamBand"], df["CompetencyText"]).to_string())

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

    # Random Forest tuning. SMOTE is applied INSIDE each CV fold via the pipeline below --
    # doing SMOTE once before RandomizedSearchCV lets synthetic points leak across folds,
    # biasing the search toward unregularized, overfit configurations. The search space is
    # also capped to sane regularization ranges as an extra safeguard.
    print("\n=== RANDOM FOREST TUNING (regularized, SMOTE applied inside CV) ===")
    rf_pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("rf", RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", bootstrap=True, n_jobs=-1)),
    ])
    rf_param_grid = {
        "rf__n_estimators": [200, 300, 500],
        "rf__max_depth": [4, 6, 8, 10],
        "rf__min_samples_split": [20, 40, 60],
        "rf__min_samples_leaf": [10, 20, 30],
        "rf__max_features": ["sqrt", "log2"],
    }
    rf_search = RandomizedSearchCV(
        estimator=rf_pipeline,
        param_distributions=rf_param_grid,
        n_iter=20,
        scoring="f1_weighted",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    rf_search.fit(X_train_processed, y_train)  # raw (pre-SMOTE); SMOTE happens per-fold inside the pipeline
    best_rf_params = {k.replace("rf__", ""): v for k, v in rf_search.best_params_.items()}
    print("Best RF parameters:", best_rf_params)
    print("Best RF CV weighted F1:", round(float(rf_search.best_score_), 4))

    # Refit the selected configuration on the full SMOTE-balanced training set
    # (no CV here, so this refit cannot leak into validation/test).
    rf_model = RandomForestClassifier(
        random_state=RANDOM_STATE, class_weight="balanced", bootstrap=True, n_jobs=-1, **best_rf_params
    )
    rf_model.fit(X_train_balanced, y_train_balanced)

    # Hybrid RF-DNN input creation. OOF RF probabilities are computed on the RAW (pre-SMOTE)
    # training rows, with SMOTE applied inside each CV fold -- running cross_val_predict on
    # already-SMOTE'd data would let synthetic points leak across folds the same way as above.
    print("\n=== HYBRID RF-DNN TUNING ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rf_oof_pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("rf", RandomForestClassifier(
            random_state=RANDOM_STATE, class_weight="balanced", bootstrap=True, n_jobs=-1, **best_rf_params
        )),
    ])
    rf_oof_probs_raw = cross_val_predict(
        rf_oof_pipeline, X_train_processed, y_train,
        cv=skf, method="predict_proba", n_jobs=-1,
    )
    rf_val_probs = rf_model.predict_proba(X_val_processed)
    rf_test_probs = rf_model.predict_proba(X_test_processed)

    # Build the hybrid feature set from the raw (pre-SMOTE) rows + their honest OOF RF
    # probabilities, then SMOTE the combined representation once for the final MLP fit
    # (safe here since there is no CV split happening at this point).
    X_train_hybrid_raw = np.hstack([to_dense(X_train_processed), rf_oof_probs_raw])
    smote_hybrid = SMOTE(random_state=RANDOM_STATE)
    X_train_hybrid, y_train_hybrid = smote_hybrid.fit_resample(X_train_hybrid_raw, y_train)

    X_val_hybrid = np.hstack([to_dense(X_val_processed), rf_val_probs])
    X_test_hybrid = np.hstack([to_dense(X_test_processed), rf_test_probs])

    print(
        f"Hybrid input dimension: {X_train_hybrid.shape[1]} "
        f"({to_dense(X_train_processed).shape[1]} profile features + 3 RF probabilities)"
    )

    # Heavier L2 regularization (alpha) and smaller networks than a naive search would try --
    # with a strong honest signal (banded ExamScore) available, a small/regularized network
    # reaches good accuracy without needing a jagged, unstable decision surface.
    search_configs = [
        {"arch": (32,),      "alpha": 1e-2, "lr": 1e-3, "batch": 64},
        {"arch": (64,),      "alpha": 1e-2, "lr": 1e-3, "batch": 64},
        {"arch": (64, 32),   "alpha": 1e-2, "lr": 1e-3, "batch": 64},
        {"arch": (32,),      "alpha": 5e-2, "lr": 1e-3, "batch": 64},
        {"arch": (64,),      "alpha": 5e-2, "lr": 1e-3, "batch": 64},
        {"arch": (64, 32),   "alpha": 5e-2, "lr": 1e-3, "batch": 64},
        {"arch": (32,),      "alpha": 1e-1, "lr": 1e-3, "batch": 64},
        {"arch": (64,),      "alpha": 1e-1, "lr": 1e-3, "batch": 64},
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
        model.fit(X_train_hybrid, y_train_hybrid)
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
            f"arch={str(cfg['arch']):10s} alpha={cfg['alpha']} "
            f"lr={cfg['lr']} batch={cfg['batch']} -> val F1={val_f1:.4f}"
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_hybrid_model = model
            best_architecture = cfg["arch"]
            best_hybrid_config = cfg

    hybrid_search_df = pd.DataFrame(hybrid_search_records).sort_values("Validation F1", ascending=False)
    hybrid_search_df.to_csv(MODEL_DIR / "hybrid_tuning_results.csv", index=False)

    print("Selected architecture:", best_architecture)
    print("Selected alpha:", best_hybrid_config["alpha"])
    print("Selected learning rate:", best_hybrid_config["lr"])
    print("Selected batch size:", best_hybrid_config["batch"])
    print("Best validation F1:", round(float(best_val_f1), 4))

    # Raw MLP softmax output on this data is badly miscalibrated: predict_proba routinely
    # reports 99%+ confidence for a class regardless of true accuracy. Wrap the selected
    # (already-fitted) MLP with sigmoid calibration fit on the untouched validation set, so
    # predict_proba reflects real-world reliability instead of raw softmax certainty. This
    # does not touch the test set, so the test metrics below stay honest.
    hybrid_model = CalibratedClassifierCV(FrozenEstimator(best_hybrid_model), method="sigmoid")
    hybrid_model.fit(X_val_hybrid, y_val)

    # Evaluation
    rf_val_pred = rf_model.predict(X_val_processed)
    rf_test_pred = rf_model.predict(X_test_processed)
    hybrid_val_pred = hybrid_model.predict(X_val_hybrid)
    hybrid_test_pred = hybrid_model.predict(X_test_hybrid)

    results = []
    for dataset_name, y_true, preds in [
        ("Validation", y_val, {"Random Forest": rf_val_pred, "Hybrid RF-DNN": hybrid_val_pred}),
        ("Test", y_test, {"Random Forest": rf_test_pred, "Hybrid RF-DNN": hybrid_test_pred}),
    ]:
        for model_name, pred in preds.items():
            results.append(evaluate_model(model_name, y_true, pred, dataset_name))

    results_df = pd.DataFrame(results)
    print("\n=== MODEL COMPARISON ===")
    print(results_df.to_string(index=False))
    print(
        "NOTE: the Hybrid RF-DNN 'Validation' row above is no longer an independent check -- "
        "that split was reused to fit probability calibration. The 'Test' row is the honest, "
        "untouched estimate of real-world performance."
    )
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

    check_stability(preprocessor, rf_model, hybrid_model)

    shap_cases, shap_global_importance = run_shap_analysis(
        preprocessor, rf_model, hybrid_model, X_train_hybrid, X_test_hybrid, hybrid_test_pred, LEVEL_DECODING
    )

    summary = {
        "final_model": "Hybrid RF-DNN",
        "feature_cols": FEATURE_COLS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "level_encoding": LEVEL_ENCODING,
        "level_decoding": LEVEL_DECODING,
        "best_rf_params": best_rf_params,
        "best_hybrid_architecture": str(best_architecture),
        "best_hybrid_config": best_hybrid_config,
        "best_hybrid_validation_f1": float(best_val_f1),
        "results": results,
        "exam_score_note": (
            f"ExamScore correlates {round(float(corr_value), 4)} with raw FinalGrade, so it is "
            "banded into 4 coarse categories (<=40, 41-60, 61-80, 81-100) rather than used as a "
            "precise value. This keeps a strong honest signal without making the model a trivial "
            "lookup, and leaves the two bands that straddle a label boundary genuinely ambiguous "
            "so the behavioral features have real influence there."
        ),
        "calibration_note": (
            "hybrid_model is a CalibratedClassifierCV (sigmoid) wrapping the tuned MLP, fit on the "
            "validation split, because the raw MLP softmax output was badly overconfident. The "
            "'Validation' row above is therefore not an independent estimate; use 'Test'."
        ),
    }
    with open(MODEL_DIR / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    bundle = {
        "model_type": "Hybrid RF-DNN",
        "preprocessor": preprocessor,
        "rf_model": rf_model,
        "hybrid_model": hybrid_model,
        "feature_cols": FEATURE_COLS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "level_encoding": LEVEL_ENCODING,
        "level_decoding": LEVEL_DECODING,
        "smote_used": True,
        "best_rf_params": best_rf_params,
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
    print("- models/shap_case_studies.json")
    print("- models/shap_global_importance.csv")


if __name__ == "__main__":
    main()
