# 🚗 CarPriceML - Prédiction Prix Voitures d'Occasion

![Python](https://img.shields.io/badge/Python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green) ![Docker](https://img.shields.io/badge/Docker-Ready-blue)

## 📋 Description

Système MLOps complet pour prédire le prix des voitures au Maroc avec ML, API REST, cache Redis, et monitoring Prometheus/Grafana.

**Features :** Random Forest + Transformation Log | FastAPI + Redis | Streamlit | Prometheus + Grafana | Docker (5 services)

## 🚀 Installation Rapide
```bash
# 1. Cloner et préparer
git clone <repo>
cd CarPriceML
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Entraîner le modèle
cd pipeline
python train.py
cd ..

# 3. Lancer avec Docker
docker-compose up --build -d

# 4. Accès
# Frontend:  http://localhost:8501
# API:       http://localhost:8000
# Docs:      http://localhost:8000/docs
# Grafana:   http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

## 📊 Structure
```
CarPriceML/
├── data/              # Dataset CSV
├── pipeline/          # train.py, explore.py
├── models/            # Modèles .joblib
├── app/               # Backend FastAPI
├── frontend/          # Interface Streamlit
├── monitoring/        # Config Prometheus/Grafana
├── tests/             # Tests Pytest
└── docker-compose.yml # 5 services orchestrés
```

## 🔌 API

**Prédiction :**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"year": 2014, "max_power_bhp": 74, "torque_nm": 190, "engine_cc": 1248}'
```

**Réponse :**
```json
{
  "predicted_price": 123456.78,
  "currency": "MAD",
  "model_version": "v1.0",
  "cached": false,
  "prediction_id": "abc123",
  "timestamp": "2025-11-04T12:30:00"
}
```

**Métriques :** `GET /metrics` (Prometheus)  
**Health :** `GET /health`

## ⚡ Performance

- **R² train :** 0.928
- **R² test :** 0.868
- **RMSE test :** 45417.64
- **MAE test :** 31670.03

## 📈 Monitoring Grafana

1. http://localhost:3000 (admin/admin)
2. Créer Dashboard → Add visualization → Prometheus
3. Métrique : `predictions_total`
4. Apply → Save

**Métriques disponibles :**
- `predictions_total` - Total prédictions
- `cache_hits_total` - Cache hits
- `prediction_duration_seconds` - Latence
- `model_loaded` - État modèle

## 🐛 Dépannage

**Modèle non chargé :**
```bash
pip install scikit-learn==1.4.0
cd pipeline && python train.py && cd ..
docker-compose restart backend
```

**Redis non connecté :**
```bash
docker-compose restart redis
```

**Services :**
```bash
docker-compose ps        # État
docker-compose logs      # Logs
docker-compose down      # Arrêter
```

## 🧪 Tests
```bash
pytest tests/ -v
```

## 🔧 Stack Technique

Python 3.11 | Scikit-learn 1.4.0 | FastAPI | Streamlit | Redis | Prometheus | Grafana | Docker

