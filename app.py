from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import joblib
import os

from functionalities import prepare_data, train_model, save_model

from elasticsearch import Elasticsearch
import datetime

# === Chemin du modèle
MODEL_PATH = "gb_model.pkl"

# === Charger le modèle et le scaler au démarrage
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Modèle non trouvé à : {MODEL_PATH}")

model_bundle = joblib.load(MODEL_PATH)
if not isinstance(model_bundle, dict) or "model" not in model_bundle or "scaler" not in model_bundle:
    raise ValueError(f"❌ Le fichier chargé n'est pas un format valide (dict avec 'model' et 'scaler') : {MODEL_PATH}")

model = model_bundle["model"]
scaler = model_bundle["scaler"]

# === Connexion à Elasticsearch
def connect_elasticsearch():
    # version corrigée que je t'ai donnée
    es = Elasticsearch(
        "http://localhost:9200",
        request_timeout=30,
        verify_certs=False,
        ssl_show_warn=False
    )
    if es.ping():
        print("✅ Elasticsearch connecté")
    else:
        print("❌ Impossible de se connecter à Elasticsearch (ping=False)")
    return es

es_client = connect_elasticsearch()


# === Initialiser l’application FastAPI
app = FastAPI(
    title="API Prédiction GHG",
    version="1.0",
    description="""
    API pour prédire les émissions de GES (GHG) à partir de variables économiques et énergétiques,
    et pour réentraîner le modèle sur de nouvelles données.
    """
)

# === Structure des données d’entrée
class PredictionInput(BaseModel):
    value_added: float = Field(..., description="Valeur ajoutée en millions d'euros (Value Added [M.EUR])")
    employment: float = Field(..., description="Nombre d'employés en milliers (Employment [1000 p.])")
    energy: float = Field(..., description="Consommation totale d'énergie en TJ (Energy Carrier Net Total [TJ])")
    year: int = Field(..., description="Année de la mesure")

# === Endpoint de prédiction
@app.post("/predict", summary="Faire une prédiction GHG", tags=["Prédiction"])
def predict(data: PredictionInput):
    try:
        input_array = np.array([[data.value_added, data.employment, data.energy, data.year]])
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)
        pred_value = float(prediction[0])

        # Log dans Elasticsearch
        if es_client.ping():
            doc = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "input": data.dict(),
                "prediction": pred_value
            }
            es_client.index(index="mlflow-metrics", document=doc)

        return {"prediction": pred_value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Endpoint de réentraînement
@app.post("/retrain", summary="Réentraîner le modèle avec les nouvelles données", tags=["Réentraînement"])
async def retrain():
    global model
    global scaler
    try:
        X_train_scaled, X_test_scaled, y_train, y_test, new_scaler = prepare_data("projet5.csv")
        new_model = train_model(X_train_scaled, y_train)
        model_bundle = {
            "model": new_model,
            "scaler": new_scaler
        }
        save_model(model_bundle)
        model = new_model
        scaler = new_scaler

        return {"message": "✅ Modèle réentraîné et mis à jour avec succès."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du réentraînement : {str(e)}")
