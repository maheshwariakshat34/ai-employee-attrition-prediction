import os
from flask import Flask
from routes.prediction import prediction_bp
from routes.auth import auth_bp
from database.db import init_db


app = Flask(__name__)

init_db(app)

app.register_blueprint(prediction_bp)

app.register_blueprint(auth_bp)
if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode)
