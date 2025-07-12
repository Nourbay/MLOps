import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from functionalities import (add_features, clean_data, prepare_model_data,
                             scale_data)


def load_data(filepath):
    print("1. 📥 Chargement des données...")
    df = pd.read_csv(filepath)
    print(f"-> Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes.")
    return df


def clean_and_feature_engineer(df):
    print("2. 🧹 Nettoyage et création de variables...")
    df_clean = clean_data(df)
    df_feat = add_features(df_clean)
    print(
        f"-> Données après traitement : {df_feat.shape[0]} lignes, {df_feat.shape[1]} colonnes."
    )
    return df_feat


def prepare_and_split(df):
    print("3. ✂️ Séparation des variables et split train/test...")
    X, y = prepare_model_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(
        f"-> Train : {X_train.shape[0]} échantillons, Test : {X_test.shape[0]} échantillons."
    )
    return X_train, X_test, y_train, y_test


def scale(X_train, X_test):
    print("4. ⚖️ Standardisation...")
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    print("-> Standardisation terminée.")
    return X_train_scaled, X_test_scaled, scaler


def train_model(X_train_scaled, y_train, hyperparams):
    print("5. 🚀 Entraînement du modèle Gradient Boosting...")
    model = GradientBoostingRegressor(**hyperparams)
    model.fit(X_train_scaled, y_train)
    print("-> Modèle entraîné.")
    return model


def evaluate_model(model, X_test_scaled, y_test):
    print("6. 🧪 Évaluation du modèle...")
    y_pred = model.predict(X_test_scaled)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f"✅ RMSE: {rmse:.4f}")
    print(f"✅ R2: {r2:.4f}")

    return {"rmse": rmse, "r2": r2}, y_pred


def save_model(model, scaler, path="gb_model.pkl"):
    print("7. 💾 Sauvegarde du modèle localement...")
    joblib.dump({"model": model, "scaler": scaler}, path)
    print(f"-> Modèle sauvegardé sous : {path}")


def main():
    # Chemin de données
    filepath = "projet5.csv"
    model_output_path = "gb_model.pkl"

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
        print("\n=== 📈 MLflow Tracking commencé ===")

        # Log des hyperparamètres
        mlflow.log_params(hyperparams)

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

        # Sauvegarde locale
        save_model(model, scaler, path=model_output_path)

        # Log du modèle dans MLflow
        mlflow.sklearn.log_model(model, "model")
        print("✅ Modèle et métriques enregistrés dans MLflow !")
        print("=== 🏁 MLflow Tracking terminé ===\n")


if __name__ == "__main__":
    main()
