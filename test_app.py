# Import Flask app
from app import app

# Test home route
def test_home():
    client = app.test_client()
    response = client.get("/")

    assert response.status_code == 200
    assert b"Flask ML App is running successfully" in response.data

# Test prediction route
def test_predict():
    client = app.test_client()

    response = client.post("/predict", json={"feature": 7})
    data = response.get_json()

    assert response.status_code == 200
    assert round(data["prediction"], 2) == 14.00