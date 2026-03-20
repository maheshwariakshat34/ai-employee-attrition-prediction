from app import app
from database.db import db
from database.models import User

with app.app_context():
    db.create_all()
    print("Tables created")
