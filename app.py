import os
from flask import Flask, redirect, url_for
from database.db import init_db,db
from routes.auth import auth_bp
from routes.prediction import prediction_bp

app = Flask(__name__)

# Secret key is required for session
app.secret_key = "your_secret_key_here"

# Connect the database to the app
init_db(app)

# Register all blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(prediction_bp)
with app.app_context():
    db.create_all()


# Opening the app always starts at signup page
@app.route("/")
def root():
    return redirect(url_for("auth_bp.signup"))


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode)