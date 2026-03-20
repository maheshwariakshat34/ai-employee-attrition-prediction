from flask import Blueprint, request, jsonify, redirect, url_for, render_template
from services.user_service import create_user, user_exists

auth_bp = Blueprint("auth_bp", __name__)


@auth_bp.route("/signup", methods=["GET", "POST"])
def signup():

    if request.method == "GET":
        return render_template("signup.html")

    # Handle the form submission
    if request.method == "POST":

        username    = request.form.get("username")
        email       = request.form.get("email")
        password    = request.form.get("password")
        company     = request.form.get("company")
        designation = request.form.get("designation")
        experience  = request.form.get("experience")


        if not username or not email or not password:
            return jsonify({"success": False, "message": "Username, email and password are required"}), 400

        if user_exists(email):
            return jsonify({"success": False, "message": "User already exists. Please log in."}), 409

        create_user(username, email, password, company, designation, experience)

        return redirect(url_for("auth_bp.login"))


@auth_bp.route("/login", methods=["GET"])
def login():
    # Placeholder — login logic will be added separately
    return render_template("login.html")