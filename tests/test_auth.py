from app.services.user_services import pwd_context


def test_password_hash_round_trip():
    hashed = pwd_context.hash("correct horse battery staple")

    assert pwd_context.verify("correct horse battery staple", hashed)
    assert not pwd_context.verify("wrong password", hashed)
