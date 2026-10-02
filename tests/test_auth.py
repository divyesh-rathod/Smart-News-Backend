import asyncio

from fastapi.testclient import TestClient

from app.main import app
from app.services.user_services import pwd_context
from app.utils.auth import create_access_token
from tests.factories import add_user


def test_password_hash_round_trip():
    hashed = pwd_context.hash("correct horse battery staple")

    assert pwd_context.verify("correct horse battery staple", hashed)
    assert not pwd_context.verify("wrong password", hashed)


def test_the_docs_ask_for_a_bearer_token():
    # Swagger's Authorize dialog follows this scheme. An OAuth2 password flow would post a form to the
    # login route, which takes JSON.
    schemes = app.openapi()["components"]["securitySchemes"]

    assert any(s["type"] == "http" and s["scheme"] == "bearer" for s in schemes.values())
    assert all(s["type"] != "oauth2" for s in schemes.values())


def test_protected_routes_reject_requests_without_a_valid_token():
    client = TestClient(app)

    missing = client.get("/api/V1/news/unseen-articles")
    forged = client.get("/api/V1/news/unseen-articles", headers={"Authorization": "Bearer not-a-real-token"})

    assert missing.status_code == 401
    assert missing.headers["WWW-Authenticate"] == "Bearer"
    assert forged.status_code == 401


def test_a_token_from_login_opens_protected_routes(db):
    user = asyncio.run(add_user())
    token = asyncio.run(create_access_token({"user_id": str(user.id)}))

    response = TestClient(app).get("/api/V1/news/unseen-articles", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    assert response.json() == {"results": [], "next_cursor": None}
