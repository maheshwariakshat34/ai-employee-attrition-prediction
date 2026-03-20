import os
import joblib
import shap
import pandas as pd



FEATURE_NAMES = [
    "OverTime_Yes",
    "MaritalStatus_Single",
    "TotalWorkingYears",
    "JobLevel",
    "YearsInCurrentRole",
    "MonthlyIncome",
    "Age",
    "JobRole_Sales Representative",
    "YearsWithCurrManager",
    "StockOptionLevel"
]


model_path = os.path.join("models", "employee_attrition_model.pkl")
model      = joblib.load(model_path)
explainer  = shap.TreeExplainer(model)


# ── Public API ────────────────────────────────────────────────────────────────

def predict_employee(form_data) -> dict:
    """
    Run the full ML pipeline for one employee record.

    Parameters
    ----------
    form_data : dict-like (e.g. flask.request.form)
        Validated form values.

    Returns
    -------
    dict with keys:
        prediction      – int  (0 = stays, 1 = leaves)
        attrition_prob  – float, % chance of leaving
        retention_prob  – float, % chance of staying
        top_features    – list of top-5 SHAP driver dicts
    """
    input_df       = _build_input_dataframe(form_data)
    prediction, attrition_prob, retention_prob = _run_inference(input_df)
    top_features   = _compute_shap_features(input_df)

    return {
        "prediction"    : prediction,
        "attrition_prob": attrition_prob,
        "retention_prob": retention_prob,
        "top_features"  : top_features,
    }


# ── Private helpers ───────────────────────────────────────────────────────────

def _build_input_dataframe(form_data) -> pd.DataFrame:
    """Coerce raw form strings into correct types and build a model-ready DataFrame."""
    record = {
        "OverTime_Yes"                : 1 if form_data.get("OverTime_Yes") == "1" else 0,
        "MaritalStatus_Single"        : 1 if form_data.get("MaritalStatus_Single") == "1" else 0,
        "TotalWorkingYears"           : float(form_data["TotalWorkingYears"]),
        "JobLevel"                    : int(form_data["JobLevel"]),
        "YearsInCurrentRole"          : float(form_data["YearsInCurrentRole"]),
        "MonthlyIncome"               : float(form_data["MonthlyIncome"]),
        "Age"                         : int(form_data["Age"]),
        "JobRole_Sales Representative": 1 if form_data.get("JobRole_Sales Representative") == "1" else 0,
        "YearsWithCurrManager"        : float(form_data["YearsWithCurrManager"]),
        "StockOptionLevel"            : int(form_data["StockOptionLevel"]),
    }
    return pd.DataFrame([record], columns=FEATURE_NAMES)


def _run_inference(input_df: pd.DataFrame) -> tuple:
    """Run model.predict and model.predict_proba, return prediction + probabilities."""
    prediction     = int(model.predict(input_df)[0])
    proba          = model.predict_proba(input_df)[0]
    attrition_prob = round(float(proba[1]) * 100, 2)
    retention_prob = round(float(proba[0]) * 100, 2)
    return prediction, attrition_prob, retention_prob


def _compute_shap_features(input_df: pd.DataFrame) -> list:
    """Compute SHAP values and return top-5 most influential features."""
    shap_values = explainer(input_df)

    # shap_values.values shape: (1, n_features) or (1, n_features, n_classes)
    sv = shap_values.values[0]

    # For binary classifiers that return 3D output, take class-1 slice
    if sv.ndim == 2:
        sv = sv[:, 1]

    shap_pairs = list(zip(FEATURE_NAMES, sv.tolist()))
    shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    return [
        {"feature": name, "value": round(value, 4)}
        for name, value in shap_pairs[:5]
    ]