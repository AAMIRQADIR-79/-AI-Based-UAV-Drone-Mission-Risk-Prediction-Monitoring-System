"""
ml_model.py - Machine Learning Ensemble for UAV Risk Prediction
Uses Random Forest + Gradient Boosting + Extra Trees (soft voting)
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score,
    roc_auc_score, confusion_matrix,
    classification_report,
)

from utils.config import FEATURE_NAMES
from utils.data_generator import generate_tabular_data

# ─────────────────────────────────────────────
# MODEL SAVE PATHS
# ─────────────────────────────────────────────
MODEL_DIR    = "models"
MODEL_PATH   = os.path.join(MODEL_DIR, "ensemble_model.joblib")
SCALER_PATH  = os.path.join(MODEL_DIR, "ensemble_scaler.joblib")
META_PATH    = os.path.join(MODEL_DIR, "ensemble_meta.joblib")


# ─────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────
def train_model(force_retrain: bool = False):
    """
    Train ensemble ML model or load from cache.
    Returns (model, scaler, meta_dict)
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load from cache if exists
    if (not force_retrain
            and os.path.exists(MODEL_PATH)
            and os.path.exists(SCALER_PATH)
            and os.path.exists(META_PATH)):
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        meta   = joblib.load(META_PATH)
        meta["status"] = "loaded_from_cache"
        return model, scaler, meta

    # Generate training data
    print("Generating training data...")
    df = generate_tabular_data(n_samples=8000)
    X  = df[FEATURE_NAMES].values
    y  = df["risk"].values

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Scale features
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Individual models ─────────────────────
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    et = ExtraTreesClassifier(
        n_estimators=250,
        max_depth=14,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # ── Soft voting ensemble ──────────────────
    print("Training ensemble (RF + GBM + ExtraTrees)...")
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("et", et)],
        voting="soft",
        weights=[3, 2, 2],
    )
    ensemble.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────
    y_pred  = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)[:, 1]

    acc = round(accuracy_score(y_test, y_pred)   * 100, 2)
    f1  = round(f1_score(y_test, y_pred)         * 100, 2)
    auc = round(roc_auc_score(y_test, y_proba)   * 100, 2)
    cm  = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    # Feature importances from RF
    rf_model = ensemble.named_estimators_["rf"]
    fi = dict(zip(FEATURE_NAMES, rf_model.feature_importances_))

    meta = {
        "status":              "trained",
        "accuracy":            acc,
        "f1_score":            f1,
        "auc_roc":             auc,
        "confusion_matrix":    cm,
        "report":              report,
        "feature_importance":  fi,
        "n_samples":           len(df),
        "n_train":             len(X_train),
        "n_test":              len(X_test),
        "class_balance": {
            "safe":      int((y == 0).sum()),
            "high_risk": int((y == 1).sum()),
        },
    }

    # Save everything
    joblib.dump(ensemble, MODEL_PATH)
    joblib.dump(scaler,   SCALER_PATH)
    joblib.dump(meta,     META_PATH)

    print(f"Model trained! Accuracy: {acc}% | AUC: {auc}% | F1: {f1}%")
    return ensemble, scaler, meta


# ─────────────────────────────────────────────
# PREDICT RISK
# ─────────────────────────────────────────────
def predict_risk(model, scaler, features: dict) -> dict:
    """
    Predict mission risk for given features.
    
    Args:
        features: dict with keys matching FEATURE_NAMES
    
    Returns:
        dict with risk_score, label, level, confidence
    """
    X  = np.array([[features[f] for f in FEATURE_NAMES]], dtype=float)
    Xs = scaler.transform(X)

    proba      = model.predict_proba(Xs)[0]
    risk_score = float(proba[1]) * 100

    # Determine level
    if risk_score >= 80:
        level = "Critical"
        emoji = "🚨"
        color = "#FF0000"
    elif risk_score >= 60:
        level = "High Risk"
        emoji = "🔴"
        color = "#FF6600"
    elif risk_score >= 35:
        level = "Moderate"
        emoji = "🟡"
        color = "#FFAA00"
    else:
        level = "Safe"
        emoji = "🟢"
        color = "#00CC00"

    return {
        "risk_score":  round(risk_score, 1),
        "safe_score":  round(100 - risk_score, 1),
        "label":       f"{emoji} {level}",
        "level":       level,
        "emoji":       emoji,
        "color":       color,
        "confidence":  round(max(proba) * 100, 1),
    }


# ─────────────────────────────────────────────
# FEATURE CONTRIBUTIONS (SHAP-style)
# ─────────────────────────────────────────────
def get_feature_contributions(model, scaler, features: dict) -> dict:
    """
    Calculate how much each feature contributes to the risk score.
    Uses perturbation method - change each feature by 15% and
    measure the change in risk prediction.
    """
    base_pred = predict_risk(model, scaler, features)
    base_risk = base_pred["risk_score"]

    contributions = {}
    for feat in FEATURE_NAMES:
        # Perturb this feature up by 15%
        delta = max(abs(features[feat]) * 0.15, 0.5)
        feat_up = {**features, feat: features[feat] + delta}
        risk_up = predict_risk(model, scaler, feat_up)["risk_score"]

        # Contribution = how much risk changes when feature increases
        contributions[feat] = round((risk_up - base_risk) / delta * delta, 3)

    return {
        "contributions": contributions,
        "base_risk":     base_risk,
        "feature_names": FEATURE_NAMES,
    }


# ─────────────────────────────────────────────
# GET FEATURE IMPORTANCE TABLE
# ─────────────────────────────────────────────
def get_feature_importance(model) -> pd.DataFrame:
    """Return feature importance as sorted DataFrame."""
    rf = model.named_estimators_["rf"]
    df = pd.DataFrame({
        "feature":    FEATURE_NAMES,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# GENERATE AI EXPLANATION TEXT
# ─────────────────────────────────────────────
def generate_explanation(pred: dict, features: dict,
                          contribs: dict) -> str:
    """
    Generate plain English explanation of AI prediction.
    """
    level = pred["level"]
    score = pred["risk_score"]

    # Sort features by absolute contribution
    sorted_contribs = sorted(
        contribs["contributions"].items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:3]

    labels = {
        "distance_km":          "Mission distance",
        "battery_pct":          "Battery level",
        "wind_speed_mps":       "Wind speed",
        "payload_kg":           "Payload weight",
        "signal_strength_pct":  "Signal strength",
        "temperature_c":        "Temperature",
        "altitude_m":           "Altitude",
        "humidity_pct":         "Humidity",
        "flight_time_min":      "Flight time",
        "battery_drain_rate":   "Battery drain rate",
    }

    lines = [f"**AI Assessment: {pred['label']} ({score:.1f}%)**\n"]
    lines.append("**Top 3 risk factors:**")
    for i, (feat, val) in enumerate(sorted_contribs, 1):
        direction = "increases" if val > 0 else "reduces"
        feat_val  = features[feat]
        lbl       = labels.get(feat, feat)
        lines.append(f"{i}. {lbl} ({feat_val:.1f}) → {direction} risk")

    lines.append("\n**Recommendations:**")
    if features["battery_pct"] < 30:
        lines.append("- Charge battery to at least 80% before flight")
    if features["wind_speed_mps"] > 10:
        lines.append("- Wait for wind to drop below 8 m/s")
    if features["distance_km"] > 80:
        lines.append("- Plan a battery checkpoint at midpoint")
    if features["signal_strength_pct"] < 40:
        lines.append("- Improve signal before launch")
    if level in ["Safe", "Moderate"]:
        lines.append("- ✅ Mission parameters are within acceptable limits")

    return "\n".join(lines)