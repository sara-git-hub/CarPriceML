from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import joblib
import pandas as pd
import redis
import json
import hashlib
import logging
from datetime import datetime
from typing import Optional
import os

# CONFIGURATION
MODEL_PATH = os.getenv("MODEL_PATH")
FEATURE_INFO_PATH = os.getenv("FEATURE_INFO_PATH")
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT"))
REDIS_TTL = int(os.getenv("REDIS_TTL"))
MODEL_VERSION = os.getenv("MODEL_VERSION")

# LOGGING  
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FASTAPI APP
app = FastAPI(
    title="CarPrice Prediction API",
    description="API pour prédire le prix des voitures d'occasion au Maroc",
    version=MODEL_VERSION
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REDIS
try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
        socket_connect_timeout=5
    )
    redis_client.ping()
    logger.info(f"✅ Redis connecté sur {REDIS_HOST}:{REDIS_PORT}")
except Exception as e:
    redis_client = None
    logger.warning(f"⚠️ Redis non disponible: {e}")

# CHARGER MODÈLE
try:
    model = joblib.load(MODEL_PATH)
    feature_info = joblib.load(FEATURE_INFO_PATH)
    logger.info(f"✅ Modèle chargé - Version: {MODEL_VERSION}")
except Exception as e:
    logger.error(f"❌ Erreur chargement modèle: {e}")
    model = None
    feature_info = None

# MÉTRIQUES PROMETHEUS
# Compteurs
predictions_total = Counter(
    'predictions_total',
    'Nombre total de prédictions'
)
cache_hits = Counter(
    'cache_hits_total',
    'Nombre de hits cache Redis'
)
cache_misses = Counter(
    'cache_misses_total',
    'Nombre de miss cache Redis'
)
errors_total = Counter(
    'errors_total',
    'Nombre total d\'erreurs',
    ['error_type']
)

# Histogramme pour latence
prediction_duration = Histogram(
    'prediction_duration_seconds',
    'Durée des prédictions en secondes'
)

# Gauges pour état système
model_loaded = Gauge(
    'model_loaded',
    'Modèle chargé (1) ou non (0)'
)
redis_connected = Gauge(
    'redis_connected',
    'Redis connecté (1) ou non (0)'
)

# Initialiser les gauges
model_loaded.set(1 if model else 0)
redis_connected.set(1 if redis_client else 0)

# MODÈLES PYDANTIC
class CarFeatures(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "year": 2014,
                    "max_power_bhp": 74,
                    "torque_nm": 190,
                    "engine_cc": 1248
                }
            ]
        }
    }
    
    year: int = Field(..., ge=1990, le=2025, description="Année de fabrication")
    max_power_bhp: int = Field(..., ge=0, description="Puissance maximale (chevaux)")
    torque_nm: int = Field(..., ge=0, description="Couple moteur (Nm)")
    engine_cc: int = Field(..., ge=0, description="Cylindrée du moteur (cm³)")

class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    predicted_price: float = Field(..., description="Prix prédit en MAD")
    currency: str = "MAD"
    input_features: dict
    model_version: str
    cached: bool
    prediction_id: str
    timestamp: str

# FONCTIONS UTILITAIRES
def generate_cache_key(features: dict) -> str:
    """Générer une clé de cache unique basée sur les features"""
    features_str = json.dumps(features, sort_keys=True)
    return f"prediction:{hashlib.md5(features_str.encode()).hexdigest()}"

def generate_prediction_id() -> str:
    """Générer un ID unique pour la prédiction"""
    timestamp = datetime.now().isoformat()
    return hashlib.sha256(timestamp.encode()).hexdigest()[:16]

def save_prediction_log(prediction_data: dict):
    """Sauvegarder la prédiction dans Redis pour traçabilité"""
    if not redis_client:
        return
    
    try:
        log_key = f"log:{prediction_data['prediction_id']}"
        redis_client.setex(
            log_key,
            REDIS_TTL * 24,  # Garder les logs 24h
            json.dumps(prediction_data)
        )
    except Exception as e:
        logger.warning(f"Erreur sauvegarde log: {e}")

# ENDPOINTS
@app.get("/", tags=["Root"])
async def root():
    """Point d'entrée principal de l'API"""
    return {
        "message": "Bienvenue sur l'API CarPrice Prediction",
        "version": MODEL_VERSION,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Vérifier l'état de santé du service"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": MODEL_VERSION,
        "features_loaded": feature_info is not None,
        "redis_connected": redis_client is not None
    }

@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Endpoint pour Prometheus - expose les métriques"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_price(car: CarFeatures):
    """
    Prédire le prix d'une voiture d'occasion

    - **year**: Année de fabrication (entre 1990 et 2025)
    - **max_power_bhp**: Puissance maximale (chevaux)
    - **torque_nm**: Couple moteur (Nm)
    - **engine_cc**: Cylindrée du moteur (cm³)
    """
    if model is None:
        errors_total.labels(error_type='model_not_loaded').inc()
        raise HTTPException(status_code=503, detail="Le modèle n'est pas disponible")

    try:
        predictions_total.inc()
        car_dict = car.dict()
        cache_key = generate_cache_key(car_dict)
        cached = False

        with prediction_duration.time():  # <<-- Mesure de latence Prometheus
            # Vérifier le cache Redis
            if redis_client:
                try:
                    cached_result = redis_client.get(cache_key)
                    if cached_result:
                        cache_hits.inc()
                        cached = True
                        result = json.loads(cached_result)
                        logger.info(f"✅ Cache HIT pour {car.year}")
                        return PredictionResponse(**result)
                except Exception as e:
                    logger.warning(f"Erreur lecture cache: {e}")

            # Cache MISS - Créer les features
            cache_misses.inc()
            vehicle_age = 2025 - car.year
            input_data = pd.DataFrame([{
                'vehicle_age': float(vehicle_age),
                'year': int(car.year),
                'max_power_bhp': float(car.max_power_bhp),
                'torque_nm': float(car.torque_nm),
                'engine_cc': float(car.engine_cc)
            }])

            # Prédiction
            prediction = model.predict(input_data)[0]
            predicted_price = round(float(prediction), 2)

            # Générer l'ID et timestamp
            prediction_id = generate_prediction_id()
            timestamp = datetime.now().isoformat()

            # Préparer la réponse
            response_data = {
                "predicted_price": predicted_price,
                "currency": "MAD",
                "input_features": car_dict,
                "model_version": MODEL_VERSION,
                "cached": cached,
                "prediction_id": prediction_id,
                "timestamp": timestamp
            }

            # Sauvegarder dans Redis
            if redis_client:
                try:
                    redis_client.setex(cache_key, REDIS_TTL, json.dumps(response_data))
                    logger.info("💾 Résultat mis en cache")
                except Exception as e:
                    logger.warning(f"Erreur écriture cache: {e}")

            # Sauvegarder le log
            save_prediction_log(response_data)
            logger.info(f"✅ Prédiction: {predicted_price} MAD pour {car.year}")

        return PredictionResponse(**response_data)

    except Exception as e:
        errors_total.labels(error_type='prediction_error').inc()
        logger.error(f"❌ Erreur prédiction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")
    
@app.get("/prediction-logs/{prediction_id}", tags=["Logging"])
async def get_prediction_log(prediction_id: str):
    """Récupérer le log d'une prédiction spécifique"""
    if not redis_client:
        raise HTTPException(
            status_code=503,
            detail="Redis non disponible - logs indisponibles"
        )
    
    try:
        log_key = f"log:{prediction_id}"
        log_data = redis_client.get(log_key)
        
        if not log_data:
            raise HTTPException(
                status_code=404,
                detail="Log de prédiction introuvable"
            )
        
        return json.loads(log_data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur récupération log: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
