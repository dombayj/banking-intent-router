"""Checks for the /classify server. Run: pytest tests/ (needs the seed 44 model in models/)."""

from fastapi.testclient import TestClient

from app.main import THRESHOLD, app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_demo_page():
    res = client.get("/")
    assert res.status_code == 200 and "Classify" in res.text


def test_clear_message_is_handled_automatically():
    r = client.post("/classify", json={"text": "I still haven't received my new card"}).json()
    assert r["intent"] == "card_arrival"
    assert r["route"] == "auto" and r["confidence"] >= THRESHOLD
    assert len(r["alternatives"]) == 2
    assert r["confidence"] >= r["alternatives"][0]["confidence"] >= r["alternatives"][1]["confidence"]


def test_route_follows_threshold():
    r = client.post("/classify", json={"text": "What's the weather like in London tomorrow?"}).json()
    assert r["route"] == ("auto" if r["confidence"] >= THRESHOLD else "human")


def test_rejects_empty_and_too_long_messages():
    assert client.post("/classify", json={"text": ""}).status_code == 422
    assert client.post("/classify", json={"text": "a" * 501}).status_code == 422
