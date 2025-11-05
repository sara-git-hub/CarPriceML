# 🚗 CarPriceML - Prédiction Prix Voitures d'Occasion

![Python](https://img.shields.io/badge/Python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green) ![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red) ![Docker](https://img.shields.io/badge/Docker-Ready-blue) ![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Description

Système MLOps complet de prédiction de prix des voitures d'occasion au Maroc. Ce projet intègre machine learning, API REST, cache intelligent, et monitoring en temps réel.

**Stack technique :** Random Forest avec transformation log | FastAPI + Redis | Streamlit | Prometheus + Grafana | Docker (5 services)

---

## ✨ Fonctionnalités

- 🤖 **ML Pipeline** : Random Forest optimisé avec R² = 86.8%
- 🚀 **API REST** : FastAPI avec validation Pydantic
- 💾 **Cache Redis** : <10ms de latence (hit rate ~80%)
- 📊 **Monitoring** : Prometheus + Grafana temps réel
- 🎨 **Interface Web** : Streamlit responsive
- ✅ **Tests** : 11 tests unitaires (100% pass)
- 🐳 **Docker** : Déploiement en un clic
- 🔒 **Sécurité** : Variables d'environnement (.env)

---

## 🚀 Installation Rapide (5 minutes)

### Prérequis
- Python 3.11+
- Docker & Docker Compose
- Git

### Étapes
```bash
# 1. Cloner le projet
git clone https://github.com/sara-git-hub/CarPriceML.git
cd CarPriceML

# 2. Copier le fichier d'environnement
cp .env.example .env
# Éditez .env et changez les mots de passe si nécessaire

# 3. Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 4. Entraîner le modèle
cd pipeline
python train.py
cd ..

# 5. Lancer tous les services
docker-compose up -d

# 6. Vérifier que tout fonctionne
docker-compose ps
```

### Accès aux Services

| Service | URL |
|---------|-----|
| 🎨 Frontend | http://localhost:8501 |
| 🔌 API | http://localhost:8000 |
| 📚 API Docs | http://localhost:8000/docs |
| 📈 Grafana | http://localhost:3000 |
| 📊 Prometheus | http://localhost:9090 |

---

## 📊 Architecture
```
┌─────────────────┐
│   Streamlit     │  ← Interface utilisateur
│   :8501         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────┐
│   FastAPI       │────►│   Redis     │  Cache (1h TTL)
│   :8000         │     │   :6379     │
└────────┬────────┘     └─────────────┘
         │
         ├──────────────┐
         │              │
         ▼              ▼
┌─────────────┐  ┌─────────────┐
│ Prometheus  │  │  Grafana    │  Monitoring
│   :9090     │  │   :3000     │
└─────────────┘  └─────────────┘
```

### Structure du Projet
```
CarPriceML/
├── .env                      # Variables d'environnement (à créer)
├── .env.example             # Template
├── .gitignore               # Fichiers ignorés par Git
├── docker-compose.yml       # Orchestration des services
├── Dockerfile               # Image backend
├── Dockerfile.frontend      # Image frontend
├── requirements.txt         # Dépendances Python
├── README.md                # Cette documentation
│
├── data/
│   └── car-details.csv     # Dataset d'entraînement
│
├── pipeline/
│   ├── train.py            # Script d'entraînement
│   ├── explore.py          # Analyse exploratoire
│   └── visualizations/     # Graphiques générés
│
├── models/
│   ├── rf_model.joblib     # Modèle entraîné
│   └── feature_info.joblib # Métadonnées du modèle
│
├── app/
│   └── main.py             # Backend FastAPI
│
├── frontend/
│   └── app.py              # Interface Streamlit
│
├── monitoring/
│   ├── prometheus.yml      # Config Prometheus
│   └── dashboards/         # Dashboards Grafana
│
└── tests/
    └── test_api.py         # Tests unitaires
```

---

## 🔌 Utilisation de l'API

### Endpoint `/predict`

**Requête :**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "year": 2014,
    "max_power_bhp": 74,
    "torque_nm": 190,
    "engine_cc": 1248
  }'
```

**Réponse :**
```json
{
  "predicted_price": 254031.34,
  "currency": "MAD",
  "input_features": {
    "year": 2014,
    "max_power_bhp": 74,
    "torque_nm": 190,
    "engine_cc": 1248
  },
  "model_version": "v1.0",
  "cached": false,
  "prediction_id": "a1b2c3d4e5f6g7h8",
  "timestamp": "2025-11-05T14:30:00.123456"
}
```

### Autres Endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/health` | GET | Vérifier l'état du service |
| `/metrics` | GET | Métriques Prometheus |
| `/` | GET | Informations générales |
| `/docs` | GET | Documentation interactive (Swagger) |

---

## 📈 Performances du Modèle

| Métrique | Train | Test |
|----------|-------|------|
| **R²** | 0.923 | 0.868 |
| **RMSE** | - | 45,418 MAD |
| **MAE** | - | 31,670 MAD |
| **Overfitting** | Δ R² = 0.055 (✅ Acceptable) |

### Graphiques Générés

Le script `train.py` génère automatiquement :
- `overfitting_analysis.png` - Comparaison train/test
- `feature_importance.png` - Top 20 variables importantes

---

## 🧪 Tests

### Lancer les Tests
```bash
# Activer l'environnement virtuel
source venv/bin/activate  # Windows: venv\Scripts\activate

# Lancer tous les tests
pytest tests/test_api.py -v

```

### Tests Inclus (11 tests)

- ✅ Endpoint racine
- ✅ Health check
- ✅ Prédiction valide
- ✅ Validation des données (année, puissance, etc.)
- ✅ Gestion des erreurs
- ✅ Format de réponse
- ✅ Métriques Prometheus
- ✅ Fonctionnalité du cache Redis

---

## 🐛 Dépannage

### Problème : Modèle non chargé

**Erreur :** `{"detail":"Modèle non chargé"}`

**Solution :**
```bash
# Vérifier la version de scikit-learn
pip show scikit-learn

# Installer la bonne version
pip install scikit-learn==1.4.0

# Réentraîner le modèle
cd pipeline
python train.py
cd ..

# Redémarrer le backend
docker-compose restart backend
```

### Problème : Redis non connecté

**Erreur :** `redis_connected: false`

**Solution :**
```bash
# Vérifier l'état de Redis
docker-compose logs redis

# Redémarrer Redis
docker-compose restart redis

# Vérifier la connexion
docker-compose exec redis redis-cli ping
```

### Problème : Port déjà utilisé

**Erreur :** `Bind for 0.0.0.0:8000 failed: port is already allocated`

**Solution :**

Modifiez `docker-compose.yml` :
```yaml
backend:
  ports:
    - "8001:8000"  # Changez 8000 en 8001
```

### Problème : Services ne démarrent pas

**Solution :**
```bash
# Voir les logs détaillés
docker-compose logs --tail 50

# Reconstruire les images
docker-compose down
docker-compose up --build -d

# Vérifier l'état
docker-compose ps
```

## 📚 Documentation Complète

- **API Documentation** : http://localhost:8000/docs (Swagger UI interactive)
- **Prometheus Queries** : http://localhost:9090/graph
- **Grafana Dashboards** : http://localhost:3000

---

## 👨‍💻 Auteur

**Sara**
- GitHub: [@sara-git-hub](https://github.com/sara-git-hub)

---
