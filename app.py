import os
import numpy as np
import pandas as pd
import joblib
import shap
from flask import Flask, request, jsonify

app = Flask(__name__)

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

@app.route("/")
def home():
    return "Employee Attrition Prediction API running"


# Route 2: Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        overtime        = 1 if request.form.get("OverTime_Yes") == "1" else 0
        marital_single  = 1 if request.form.get("MaritalStatus_Single") == "1" else 0
        sales_rep       = 1 if request.form.get("JobRole_Sales Representative") == "1" else 0

        # Read numeric fields and convert to the right type
        total_working_years   = float(request.form.get("TotalWorkingYears", 0))
        job_level             = int(request.form.get("JobLevel", 1))
        years_in_role         = float(request.form.get("YearsInCurrentRole", 0))
        monthly_income        = float(request.form.get("MonthlyIncome", 0))
        age                   = int(request.form.get("Age", 30))
        years_with_manager    = float(request.form.get("YearsWithCurrManager", 0))
        stock_option_level    = int(request.form.get("StockOptionLevel", 0))

        input_dict = {
            "OverTime_Yes"                  : overtime,
            "MaritalStatus_Single"          : marital_single,
            "TotalWorkingYears"             : total_working_years,
            "JobLevel"                      : job_level,
            "YearsInCurrentRole"            : years_in_role,
            "MonthlyIncome"                 : monthly_income,
            "Age"                           : age,
            "JobRole_Sales Representative"  : sales_rep,
            "YearsWithCurrManager"          : years_with_manager,
            "StockOptionLevel"              : stock_option_level
        }

        input_df = pd.DataFrame([input_dict], columns=FEATURE_NAMES)

        prediction = int(model.predict(input_df)[0])

        proba           = model.predict_proba(input_df)[0]
        attrition_prob  = round(float(proba[1]) * 100, 2)
        retention_prob  = round(float(proba[0]) * 100, 2)

        # SHAP explainability

        shap_values = explainer(input_df)

        # shap_values.values shape: (1, n_features) or (1, n_features, n_classes)
        sv = shap_values.values[0]

        # For binary classifiers that return 3D output, take class-1 slice
        if sv.ndim == 2:
            sv = sv[:, 1]

        # Pair each feature with its SHAP value, sort by absolute impact
        shap_pairs = list(zip(FEATURE_NAMES, sv.tolist()))
        shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)


        top_features = [
            {"feature": name, "value": round(value, 4)}
            for name, value in shap_pairs[:5]
        ]


        return jsonify({
            "success"       : True,
            "prediction"    : prediction,
            "attrition_prob": attrition_prob,
            "retention_prob": retention_prob,
            "top_features"  : top_features
        })

    except Exception as e:
        # Return the error message if something goes wrong
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
