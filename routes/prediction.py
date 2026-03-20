from flask import Blueprint, request, jsonify, render_template
from utils.validation import validate_input
from services.prediction_service import predict_employee

prediction_bp = Blueprint("prediction_bp", __name__)


@prediction_bp.route("/")
def home():
    return render_template("index.html")


@prediction_bp.route("/predict", methods=["POST"])
def predict():
    try:
        errors = validate_input(request.form)
        if errors:
            return jsonify({"success": False, "errors": errors}), 400

        result = predict_employee(request.form)

        return jsonify({"success": True, **result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
