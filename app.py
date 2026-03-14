import os
import pandas as pd
import joblib
import shap
from flask import Flask, request, jsonify ,render_template

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

REQUIRED_FIELDS = [
    "Age", "TotalWorkingYears", "YearsInCurrentRole", "YearsWithCurrManager",
    "JobLevel", "MonthlyIncome", "StockOptionLevel",
    "OverTime_Yes", "MaritalStatus_Single", "JobRole_Sales Representative"
]


def validate_input(data):
    errors = []

    for field in REQUIRED_FIELDS:
        if field not in data or data.get(field) in (None, ""):
            errors.append(f"{field!r} is required")

    if errors:
        return errors

    age = None
    total_years = None
    years_role = None
    manager_years = None

    # Age
    try:
        age = int(data.get("Age"))
        if age < 18 or age > 65:
            errors.append("Age must be between 18 and 65")
    except (TypeError, ValueError):
        errors.append("Age must be a whole number")

    # TotalWorkingYears
    try:
        total_years = float(data.get("TotalWorkingYears"))
        if total_years < 0 or total_years > 40:
            errors.append("TotalWorkingYears must be between 0 and 40")
    except (TypeError, ValueError):
        errors.append("TotalWorkingYears must be numeric")

    # YearsInCurrentRole
    try:
        years_role = float(data.get("YearsInCurrentRole"))
        if years_role < 0:
            errors.append("YearsInCurrentRole cannot be negative")
    except (TypeError, ValueError):
        errors.append("YearsInCurrentRole must be numeric")

    #YearsWithCurrManager
    try:
        manager_years = float(data.get("YearsWithCurrManager"))
        if manager_years < 0:
            errors.append("YearsWithCurrManager cannot be negative")
    except (TypeError, ValueError):
        errors.append("YearsWithCurrManager must be numeric")

    # JobLevel
    try:
        job_level = int(data.get("JobLevel"))
        if job_level < 1 or job_level > 5:
            errors.append("JobLevel must be between 1 and 5")
    except (TypeError, ValueError):
        errors.append("JobLevel must be a whole number between 1 and 5")

    # MonthlyIncome
    try:
        monthly_income = float(data.get("MonthlyIncome"))
        if monthly_income < 0:
            errors.append("MonthlyIncome cannot be negative")
    except (TypeError, ValueError):
        errors.append("MonthlyIncome must be numeric")

    # StockOptionLevel
    try:
        stock_option = int(data.get("StockOptionLevel"))
        if stock_option < 0 or stock_option > 3:
            errors.append("StockOptionLevel must be between 0 and 3")
    except (TypeError, ValueError):
        errors.append("StockOptionLevel must be a whole number between 0 and 3")

    #Cross-field logic
    if age is not None and total_years is not None:
        if total_years >= age:
            errors.append("TotalWorkingYears must be less than Age")
        if (age - total_years) < 18:
            errors.append("TotalWorkingYears is unrealistic for the given Age")

    if total_years is not None and years_role is not None:
        if years_role > total_years:
            errors.append("YearsInCurrentRole cannot exceed TotalWorkingYears")

    if years_role is not None and manager_years is not None:
        if manager_years > years_role:
            errors.append("YearsWithCurrManager cannot exceed YearsInCurrentRole")

    return errors


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        errors = validate_input(request.form)

        if errors:
            return jsonify({
                "success": False,
                "errors": errors
            }), 400

        overtime       = 1 if request.form.get("OverTime_Yes") == "1" else 0
        marital_single = 1 if request.form.get("MaritalStatus_Single") == "1" else 0
        sales_rep      = 1 if request.form.get("JobRole_Sales Representative") == "1" else 0

        total_working_years = float(request.form["TotalWorkingYears"])
        job_level           = int(request.form["JobLevel"])
        years_in_role       = float(request.form["YearsInCurrentRole"])
        monthly_income      = float(request.form["MonthlyIncome"])
        age                 = int(request.form["Age"])
        years_with_manager  = float(request.form["YearsWithCurrManager"])
        stock_option_level  = int(request.form["StockOptionLevel"])

        input_dict = {
            "OverTime_Yes"                 : overtime,
            "MaritalStatus_Single"         : marital_single,
            "TotalWorkingYears"            : total_working_years,
            "JobLevel"                     : job_level,
            "YearsInCurrentRole"           : years_in_role,
            "MonthlyIncome"                : monthly_income,
            "Age"                          : age,
            "JobRole_Sales Representative" : sales_rep,
            "YearsWithCurrManager"         : years_with_manager,
            "StockOptionLevel"             : stock_option_level
        }

        input_df = pd.DataFrame([input_dict], columns=FEATURE_NAMES)

        prediction = int(model.predict(input_df)[0])

        proba          = model.predict_proba(input_df)[0]
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
            "success"        : True,
            "prediction"     : prediction,
            "attrition_prob" : attrition_prob,
            "retention_prob" : retention_prob,
            "top_features"   : top_features
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode)