from flask import Blueprint, request, jsonify, render_template, redirect, url_for, session
from utils.validation import validate_input
from services.prediction_service import predict_employee

prediction_bp = Blueprint("prediction_bp", __name__)


@prediction_bp.route("/dashboard")
def dashboard():

    # If user is not logged in, send them to login
    if "user_id" not in session:
        return redirect(url_for("auth_bp.login"))

    # User is logged in → show the prediction form
    return render_template("index.html")


@prediction_bp.route("/predict", methods=["POST"])
def predict():

    # Protect this route too — only logged in users can predict
    if "user_id" not in session:
        return redirect(url_for("auth_bp.login"))

    try:
        errors = validate_input(request.form)
        if errors:
            return jsonify({"success": False, "errors": errors}), 400

        result = predict_employee(request.form)
        return jsonify({"success": True, **result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500