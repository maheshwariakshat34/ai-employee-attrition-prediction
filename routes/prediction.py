import os
import pandas as pd
import joblib
import shap
from flask import Blueprint, request, jsonify, render_template
from utils.validation import validate_input

prediction_bp = Blueprint("prediction_bp", __name__)

# ── Model loading (temporary home until services layer is created) ─────────────
model_path = os.path.join("models", "employee_attrition_model.pkl")
model = joblib.load(model_path)
explainer = shap.TreeExplainer(model)

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


@prediction_bp.route("/")
def home():
    return render_template("index.html")


@prediction_bp.route("/predict", methods=["POST"])
def predict():
    try:
        errors = validate_input(request.form)

        if errors:
            return jsonify({
                "success": False,
                "errors": errors
            }), 400

        overtime = 1 if request.form.get("OverTime_Yes") == "1" else 0
        marital_single = 1 if request.form.get("MaritalStatus_Single") == "1" else 0
        sales_rep = 1 if request.form.get("JobRole_Sales Representative") == "1" else 0

        total_working_years = float(request.form["TotalWorkingYears"])
        job_level = int(request.form["JobLevel"])
        years_in_role = float(request.form["YearsInCurrentRole"])
        monthly_income = float(request.form["MonthlyIncome"])
        age = int(request.form["Age"])
        years_with_manager = float(request.form["YearsWithCurrManager"])
        stock_option_level = int(request.form["StockOptionLevel"])

        input_dict = {
            "OverTime_Yes": overtime,
            "MaritalStatus_Single": marital_single,
            "TotalWorkingYears": total_working_years,
            "JobLevel": job_level,
            "YearsInCurrentRole": years_in_role,
            "MonthlyIncome": monthly_income,
            "Age": age,
            "JobRole_Sales Representative": sales_rep,
            "YearsWithCurrManager": years_with_manager,
            "StockOptionLevel": stock_option_level
        }

        input_df = pd.DataFrame([input_dict], columns=FEATURE_NAMES)

        prediction = int(model.predict(input_df)[0])

        proba = model.predict_proba(input_df)[0]
        attrition_prob = round(float(proba[1]) * 100, 2)
        retention_prob = round(float(proba[0]) * 100, 2)

        # SHAP explainability
        shap_values = explainer(input_df)

        # shap_values.values shape: (1, n_features) or (1, n_features, n_classes)
        sv = shap_values.values[0]

        # For binary classifiers that return 3D output, take class-1 slice
        if sv.ndim == 2:
            sv = sv[:, 1]

        shap_pairs = list(zip(FEATURE_NAMES, sv.tolist()))
        shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        top_features = [
            {"feature": name, "value": round(value, 4)}
            for name, value in shap_pairs[:5]
        ]

        return jsonify({
            "success": True,
            "prediction": prediction,
            "attrition_prob": attrition_prob,
            "retention_prob": retention_prob,
            "top_features": top_features
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500