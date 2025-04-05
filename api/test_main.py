from fastapi.testclient import TestClient
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.main import app  # Ajustá el path según tu estructura

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_valid():
    payload = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.02381,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_incomplete():
    payload = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        # Faltan campos
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
