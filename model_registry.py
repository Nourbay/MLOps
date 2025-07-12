#!/usr/bin/env python3

import os
import argparse
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


def load_model_artifact(model_path):
    """
    Charge un modèle local sauvegardé en .pkl ou .joblib
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Fichier modèle introuvable : {model_path}")

    print(f"📥 Chargement du modèle et du scaler depuis : {model_path}")
    bundle = joblib.load(model_path)

    model = bundle.get("model")
    scaler = bundle.get("scaler")

    if model is None:
        raise ValueError("❌ Aucun objet 'model' trouvé dans le fichier. Vérifie la sauvegarde.")

    print("✅ Modèle chargé avec succès.")
    return model, scaler


def register_to_registry(model, experiment_name, registered_model_name):
    """
    Log et enregistre un modèle scikit-learn dans le MLflow Model Registry avec description et tags
    """
    print(f"✅ Définition de l'expérience MLflow : {experiment_name}")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="Register_Model") as run:
        print(f"🚀 Nouveau run MLflow ID : {run.info.run_id}")

        print("💾 Log du modèle dans le MLflow Tracking Server...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=registered_model_name
        )
        model_uri = f"runs:/{run.info.run_id}/model"

        print("🗂️ Enregistrement dans le Model Registry...")
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name
        )

        print(f"✅ Version {model_version.version} enregistrée dans le Model Registry sous le nom : {registered_model_name}")

        # === Ajouter descriptions et tags ===
        client = MlflowClient()

        # Registered Model scope général
        client.update_registered_model(
            name=registered_model_name,
            description=(
                "Modèle GradientBoosting pour prédire des indicateurs industriels et environnementaux.\n"
                "Features utilisées : Region (catégoriel), Sector (catégoriel), "
                "Value Added [M.EUR], Employment [1000 p.], GHG emissions [kg CO2 eq.], "
                "Energy Carrier Net Total [TJ], Year."
            )
        )
        client.set_registered_model_tag(registered_model_name, "project", "IndustrialPrediction")
        client.set_registered_model_tag(registered_model_name, "author", "TonNom")
        client.set_registered_model_tag(registered_model_name, "algorithm", "GradientBoosting")
        client.set_registered_model_tag(registered_model_name, "framework", "scikit-learn")
        client.set_registered_model_tag(registered_model_name, "mlflow-version", mlflow.__version__)

        # Model Version scope version spécifique
        client.update_model_version(
            name=registered_model_name,
            version=model_version.version,
            description=(
                "Version auto-enregistrée depuis registry.app.\n"
                "Modèle scikit-learn optimisé et preprocessing éventuel pour prédiction industrielle."
            )
        )
        client.set_model_version_tag(registered_model_name, model_version.version, "stage", "Staging")
        client.set_model_version_tag(registered_model_name, model_version.version, "source", model_uri)
        client.set_model_version_tag(registered_model_name, model_version.version, "framework", "scikit-learn")
        client.set_model_version_tag(registered_model_name, model_version.version, "scaler_included", "True")
        client.set_model_version_tag(registered_model_name, model_version.version, "author", "TonNom")

        print("✅ Descriptions et tags ajoutés au modèle.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enregistre un modèle local dans le MLflow Model Registry."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Chemin vers le fichier modèle joblib/pkl (ex: mon_model.pkl)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="Industrial_Experiment",
        help="Nom de l'expérience MLflow (par défaut : Industrial_Experiment)"
    )
    parser.add_argument(
        "--registered-model-name",
        type=str,
        required=True,
        help="Nom sous lequel enregistrer le modèle dans le Model Registry"
    )

    args = parser.parse_args()

    try:
        model, scaler = load_model_artifact(args.model_path)
        register_to_registry(
            model=model,
            experiment_name=args.experiment_name,
            registered_model_name=args.registered_model_name
        )
        print("🏁 ✅ Modèle enregistré avec succès dans le MLflow Model Registry !")

    except Exception as e:
        print(f"💥 Une erreur est survenue : {e}")
