import bcrypt
from database.db import db
from database.models import User


def user_exists(email):
    email = email.strip().lower()
    user = User.query.filter_by(email=email).first()
    return user is not None


def create_user(username, email, password, company, designation, experience):

    email = email.strip().lower()

    hashed_password = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    )

    new_user = User(
        username=username,
        email=email,
        password=hashed_password,  # ✅ KEEP AS BYTES
        company=company,
        designation=designation,
        experience=experience
    )

    db.session.add(new_user)
    db.session.commit()


def authenticate_user(email, password):

    email = email.strip().lower()

    user = User.query.filter_by(email=email).first()
    if not user:
        return None

    return bcrypt.checkpw(
        password.encode("utf-8"),
        user.password
    ) and user