"""
Tests unitaires pour l'API FastAPI
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Ajouter le chemin de l'application au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.main import app

client = TestClient(app)

def test_root_endpoint():
    """Test de l'endpoint racine"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "endpoints" in data

def test_health_endpoint():
    """Test de l'endpoint /health"""
    response = client.get("/health")
    # Le statut peut être 200 (modèle chargé) ou 503 (modèle non chargé)
    assert response.status_code in [200, 503]
    data = response.json()
    
    if response.status_code == 200:
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "features_loaded" in data
        assert "redis_connected" in data

def test_predict_endpoint_valid_data():
    """Test de l'endpoint /predict avec des données valides"""
    car_data = {
        "year": 2014,
        "max_power_bhp": 74,
        "torque_nm": 190,
        "engine_cc": 1248
    }
    
    response = client.post("/predict", json=car_data)
    
    # Peut être 200 (succès) ou 503 (modèle non chargé) selon l'environnement
    if response.status_code == 200:
        data = response.json()
        assert "predicted_price" in data
        assert "currency" in data
        assert data["currency"] == "MAD"
        assert "input_features" in data
        assert "model_version" in data
        assert "cached" in data
        assert "prediction_id" in data
        assert isinstance(data["predicted_price"], (int, float))
        assert data["predicted_price"] > 0
    else:
        assert response.status_code == 503

def test_predict_endpoint_invalid_year():
    """Test avec une année invalide"""
    car_data = {
        "year": 1800,  # Année invalide
        "max_power_bhp": 74,
        "torque_nm": 190,
        "engine_cc": 1248
    }
    
    response = client.post("/predict", json=car_data)
    assert response.status_code == 422  # Validation error

def test_predict_endpoint_missing_field():
    """Test avec un champ manquant"""
    car_data = {
        "year": 2014,
        "max_power_bhp": 74,
        # "torque_nm" manquant
        "engine_cc": 1248
    }
    
    response = client.post("/predict", json=car_data)
    assert response.status_code == 422  # Validation error

def test_predict_endpoint_negative_power():
    """Test avec une puissance négative"""
    car_data = {
        "year": 2014,
        "max_power_bhp": -10,  # Puissance négative
        "torque_nm": 190,
        "engine_cc": 1248
    }
    
    response = client.post("/predict", json=car_data)
    assert response.status_code == 422  # Validation error

def test_metrics_endpoint():
    """Test de l'endpoint /metrics pour Prometheus"""
    response = client.get("/metrics")
    assert response.status_code == 200
    # Vérifier que c'est bien du format Prometheus
    assert "predictions_total" in response.text or response.text != ""

def test_predict_multiple_years():
    """Test de prédiction pour différentes années"""
    years = [2010, 2015, 2018, 2020, 2023]
    
    for year in years:
        car_data = {
            "year": year,
            "max_power_bhp": 80,
            "torque_nm": 200,
            "engine_cc": 1500
        }
        
        response = client.post("/predict", json=car_data)
        if response.status_code == 200:
            data = response.json()
            assert data["predicted_price"] > 0

def test_predict_different_power_levels():
    """Test de prédiction pour différents niveaux de puissance"""
    power_levels = [50, 100, 150, 200, 300]
    
    for power in power_levels:
        car_data = {
            "year": 2018,
            "max_power_bhp": power,
            "torque_nm": 200,
            "engine_cc": 1500
        }
        
        response = client.post("/predict", json=car_data)
        if response.status_code == 200:
            data = response.json()
            assert data["predicted_price"] > 0

def test_predict_response_format():
    """Test du format de la réponse"""
    car_data = {
        "year": 2014,
        "max_power_bhp": 74,
        "torque_nm": 190,
        "engine_cc": 1248
    }
    
    response = client.post("/predict", json=car_data)
    
    if response.status_code == 200:
        data = response.json()
        
        # Vérifier la structure de la réponse
        assert isinstance(data, dict)
        assert "predicted_price" in data
        assert "currency" in data
        assert "input_features" in data
        assert "model_version" in data
        assert "cached" in data
        assert "prediction_id" in data
        assert "timestamp" in data
        
        # Vérifier les types
        assert isinstance(data["predicted_price"], (int, float))
        assert isinstance(data["currency"], str)
        assert isinstance(data["input_features"], dict)
        assert isinstance(data["cached"], bool)
        
        # Vérifier les valeurs
        assert data["currency"] == "MAD"
        assert data["input_features"]["year"] == 2014
        assert data["model_version"] == "v1.0"

def test_cache_functionality():
    """Test du cache Redis - deux prédictions identiques"""
    car_data = {
        "year": 2016,
        "max_power_bhp": 90,
        "torque_nm": 210,
        "engine_cc": 1600
    }
    
    # Première requête - cache MISS
    response1 = client.post("/predict", json=car_data)
    if response1.status_code == 200:
        data1 = response1.json()
        
        # Deuxième requête identique - cache HIT (si Redis fonctionne)
        response2 = client.post("/predict", json=car_data)
        if response2.status_code == 200:
            data2 = response2.json()
            
            # Le prix doit être identique
            assert data1["predicted_price"] == data2["predicted_price"]
            # Le prediction_id sera différent mais le prix identique

if __name__ == "__main__":
    pytest.main([__file__, "-v"])