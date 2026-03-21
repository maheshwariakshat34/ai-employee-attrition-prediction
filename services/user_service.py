import bcrypt
from database.db import db
from database.models import User


def user_exists(email):
    user = User.query.filter_by(email=email).first()
    return user is not None  # returns True if found, False if not


def create_user(username, email, password, company, designation, experience):


    # Convert password string to bytes, then hash it
    password_bytes = password.encode("utf-8")
    hashed_password = bcrypt.hashpw(password_bytes, bcrypt.gensalt())

    # Create a new User object with the given details
    new_user = User(
        username=username,
        email=email,
        password=hashed_password,
        company=company,
        designation=designation,
        experience=experience
    )

    # Save the new user to the database
    db.session.add(new_user)
    db.session.commit()


def authenticate_user(email, password):
    user = User.query.filter_by(email=email).first()
    if user is None:
        return None
    password_matches = bcrypt.checkpw(password.encode("utf-8"), user.password)
    if password_matches:
        return user

    # If password is wrong, return None
    return None