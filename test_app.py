'''
Unit tests for API App

Author: Marina Dolokov
Date: February 2022 

'''
from fastapi.testclient import TestClient

# Import our app from main.py.
from app import app

# Instantiate the testing client with our app.
client = TestClient(app)


def test_predict():
    r = client.get("/prediction")
    assert r.status_code == 200


def test_score():
    r = client.get("/scoring")
    assert r.status_code == 200


def test_summarize():
    r = client.get("/summarystats")
    assert r.status_code == 200


def test_diagnose():
    r = client.get("/diagnostics")
    assert r.status_code == 200

    