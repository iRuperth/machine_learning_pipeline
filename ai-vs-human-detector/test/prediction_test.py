from fastapi.testclient import TestClient
from api.main import app  

client = TestClient(app)

def test_predict_endpoint():
    response = client.post("/predict?text=Este es un texto escrito por un humano")
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
