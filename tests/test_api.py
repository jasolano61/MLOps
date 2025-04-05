# tests/test_api.py
import requests

def test_predict_endpoint():
    url = "http://localhost:8000/predict"
    data = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.02381,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23
    }

    response = requests.post(url, json=data)
    
    assert response.status_code == 200
    json_data = response.json()
    assert "prediction" in json_data
    assert isinstance(json_data["prediction"], list)