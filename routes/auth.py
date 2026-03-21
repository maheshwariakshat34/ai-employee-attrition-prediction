from flask import Blueprint, request, jsonify, redirect, url_for, render_template, session
from services.user_service import create_user, user_exists, authenticate_user

auth_bp = Blueprint("auth_bp", __name__)


# ── Signup

@auth_bp.route("/signup", methods=["GET", "POST"])
def signup():
    # Show the signup form when the page is opened
    if request.method == "GET":
        return render_template("signup.html")

    # Handle form submission
    if request.method == "POST":

        # Get values from the submitted form
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        company = request.form.get("company")
        designation = request.form.get("designation")
        experience = request.form.get("experience")

        # Check if required fields are filled
        if not username or not email or not password:
            return jsonify({"success": False, "message": "Username, email and password are required"}), 400

        # Check if user already exists
        if user_exists(email):
            return jsonify({"success": False, "message": "User already exists. Please log in."}), 409

        # Create and save the new user
        create_user(username, email, password, company, designation, experience)

        # Redirect to login after successful signup
        return redirect(url_for("auth_bp.login"))


# ── Login ─────────────────────────────────────────────────────────────────────

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    # Show the login form when the page is opened
    if request.method == "GET":
        return render_template("login.html")

    # Handle form submission
    if request.method == "POST":

        # Step 1: Get email and password from the form
        email = request.form.get("email")
        password = request.form.get("password")

        # Step 2: Check if fields are filled
        if not email or not password:
            return jsonify({"success": False, "message": "Email and password are required"}), 400

        # Step 3: Check email and password against the database
        user = authenticate_user(email, password)

        # Step 4: If user is valid, save their ID in the session and go to dashboard
        if user:
            session["user_id"] = user.id  # remember who is logged in
            return redirect(url_for("prediction_bp.dashboard"))

            # Step 5: If invalid, return error message
        return jsonify({"success": False, "message": "Invalid email or password"}), 401


# ── Logout ────────────────────────────────────────────────────────────────────

@auth_bp.route("/logout")
def logout():
    # Clear everything saved in the session (forgets who is logged in)
    session.clear()

    # Send user back to login page
    return redirect(url_for("auth_bp.login"))