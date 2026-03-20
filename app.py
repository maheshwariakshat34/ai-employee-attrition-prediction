import os
from flask import Flask
from routes.prediction import prediction_bp

app = Flask(__name__)

app.register_blueprint(prediction_bp)

if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode)
