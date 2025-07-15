import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from functionalities import (add_features, clean_data, prepare_model_data,
                             scale_data)

import numpy as np
from logging_config import setup_logging, get_logger
import logging
import os
import json
from datetime import datetime



def log_mlflow_event(event_type, details):
    """Log events in a format suitable for Logstash/Elasticsearch"""
    event = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "details": details,
        "source": "mlflow_pipeline"
    }
    logger.info(f"MLFLOW_EVENT: {json.dumps(event)}")

def load_data(filepath):
    logger.info("1. 📥 Chargement des données...")
    log_mlflow_event("data_loading_start", {"filepath": filepath})
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"-> Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes.")
        log_mlflow_event("data_loading_success", {
            "rows": df.shape[0], 
            "columns": df.shape[1],
            "columns_list": list(df.columns)
        })
        return df
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {str(e)}")
        log_mlflow_event("data_loading_error", {"error": str(e)})
        raise

def clean_and_feature_engineer(df):
    logger.info("2. 🧹 Nettoyage et création de variables...")
    log_mlflow_event("data_cleaning_start", {"initial_shape": df.shape})
    
    try:
        df_clean = clean_data(df)
        df_feat = add_features(df_clean)
        logger.info(f"-> Données après traitement : {df_feat.shape[0]} lignes, {df_feat.shape[1]} colonnes.")
        log_mlflow_event("data_cleaning_success", {
            "final_shape": df_feat.shape,
            "new_features": list(set(df_feat.columns) - set(df.columns))
        })
        return df_feat
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage des données: {str(e)}")
        log_mlflow_event("data_cleaning_error", {"error": str(e)})
        raise

def prepare_and_split(df):
    logger.info("3. ✂️ Séparation des variables et split train/test...")
    log_mlflow_event("data_preparation_start", {"data_shape": df.shape})
    
    try:
        X, y = prepare_model_data(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logger.info(f"-> Train : {X_train.shape[0]} échantillons, Test : {X_test.shape[0]} échantillons.")
        log_mlflow_event("data_preparation_success", {
            "train_samples": X_train.shape[0],
            "test_samples": X_test.shape[0],
            "features": X_train.shape[1]
        })
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Erreur lors de la préparation des données: {str(e)}")
        log_mlflow_event("data_preparation_error", {"error": str(e)})
        raise

def scale(X_train, X_test):
    logger.info("4. ⚖️ Standardisation...")
    log_mlflow_event("scaling_start", {
        "train_shape": X_train.shape,
        "test_shape": X_test.shape
    })
    
    try:
        X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
        logger.info("-> Standardisation terminée.")
        log_mlflow_event("scaling_success", {
            "scaler_type": type(scaler).__name__
        })
        return X_train_scaled, X_test_scaled, scaler
    except Exception as e:
        logger.error(f"Erreur lors de la standardisation: {str(e)}")
        log_mlflow_event("scaling_error", {"error": str(e)})
        raise

def train_model(X_train_scaled, y_train, hyperparams):
    logger.info("5. 🚀 Entraînement du modèle Gradient Boosting...")
    log_mlflow_event("model_training_start", {
        "model_type": "GradientBoostingRegressor",
        "hyperparams": hyperparams,
        "train_samples": X_train_scaled.shape[0]
    })
    
    try:
        model = GradientBoostingRegressor(**hyperparams)
        model.fit(X_train_scaled, y_train)
        logger.info("-> Modèle entraîné.")
        log_mlflow_event("model_training_success", {
            "model_type": type(model).__name__,
            "feature_importance": model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else None
        })
        return model
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement: {str(e)}")
        log_mlflow_event("model_training_error", {"error": str(e)})
        raise

def evaluate_model(model, X_test_scaled, y_test):
    logger.info("6. 🧪 Évaluation du modèle...")
    log_mlflow_event("model_evaluation_start", {
        "test_samples": X_test_scaled.shape[0]
    })
    
    try:
        y_pred = model.predict(X_test_scaled)

        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"✅ RMSE: {rmse:.4f}")
        logger.info(f"✅ R2: {r2:.4f}")

        log_mlflow_event("model_evaluation_success", {
            "rmse": rmse,
            "r2": r2,
            "predictions_count": len(y_pred)
        })

        return {"rmse": rmse, "r2": r2}, y_pred
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation: {str(e)}")
        log_mlflow_event("model_evaluation_error", {"error": str(e)})
        raise

def save_model(model, scaler, path="gb_model.pkl"):
    logger.info("7. 💾 Sauvegarde du modèle localement...")
    log_mlflow_event("model_saving_start", {"path": path})
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        joblib.dump({"model": model, "scaler": scaler}, path)
        logger.info(f"-> Modèle sauvegardé sous : {path}")
        log_mlflow_event("model_saving_success", {"path": path})
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde: {str(e)}")
        log_mlflow_event("model_saving_error", {"error": str(e), "path": path})
        raise

# Enable auto-logging
mlflow.autolog()  # <-- Magic happens here!

# Setup logging
logger = setup_logging('mlflow.log', level=logging.INFO)

def main():
    # Chemin de données
    filepath = "projet5.csv"
    model_output_path = "models/gb_model.pkl"

    # Hyperparamètres
    hyperparams = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "random_state": 42,
    }

    # Définir l'expérience MLflow
    mlflow.set_experiment("GradientBoosting_Experiment")

    with mlflow.start_run(run_name="GBR_run_1"):
        logger.info("\n=== 📈 MLflow Tracking commencé ===")
        log_mlflow_event("mlflow_run_start", {
            "experiment_name": "GradientBoosting_Experiment",
            "run_name": "GBR_run_1"
        })

        # Log des hyperparamètres
        mlflow.log_params(hyperparams)
        log_mlflow_event("hyperparams_logged", hyperparams)

        try:
            # Pipeline complet
            df = load_data(filepath)
            df_prepared = clean_and_feature_engineer(df)
            X_train, X_test, y_train, y_test = prepare_and_split(df_prepared)
            X_train_scaled, X_test_scaled, scaler = scale(X_train, X_test)

            # Entraînement
            model = train_model(X_train_scaled, y_train, hyperparams)

            # Évaluation
            metrics, y_pred = evaluate_model(model, X_test_scaled, y_test)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
                log_mlflow_event("metric_logged", {metric_name: metric_value})

            # Sauvegarde locale
            save_model(model, scaler, path=model_output_path)

            # Log du modèle dans MLflow
            mlflow.sklearn.log_model(model, "model")
            logger.info("✅ Modèle et métriques enregistrés dans MLflow !")
            log_mlflow_event("mlflow_run_success", {
                "model_path": "model",
                "metrics": metrics
            })
            
        except Exception as e:
            logger.error(f"Erreur dans le pipeline ML: {str(e)}")
            log_mlflow_event("mlflow_run_error", {"error": str(e)})
            raise
        finally:
            logger.info("=== 🏁 MLflow Tracking terminé ===\n")
            log_mlflow_event("mlflow_run_end", {})

if __name__ == "__main__":
    main()
