import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import mlflow
import mlflow.sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from mlflow.models.signature import infer_signature


def clean_data(df):
    # Remplissage basique des NA
    df['sector'] = df['sector'].fillna(df['sector'].mode()[0])
    df['Value Added [M.EUR]'] = df['Value Added [M.EUR]'].fillna(df['Value Added [M.EUR]'].median())
    df['Energy Carrier Net Total [TJ]'] = df['Energy Carrier Net Total [TJ]'].fillna(df['Energy Carrier Net Total [TJ]'].median())
    df['Exp Type'] = df['Exp Type'].fillna(df['Exp Type'].median())
    return df


def add_features(df):
    # Exemple de feature simple
    df['Value_per_Employee'] = df['Value Added [M.EUR]'] / (df['Employment [1000 p.]'] + 1)
    return df


def prepare_model_data(df):
    X = df[['Value Added [M.EUR]', 'Employment [1000 p.]', 'Energy Carrier Net Total [TJ]', 'Year']]
    y = df['GHG emissions [kg CO2 eq.]']
    return X, y


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def prepare_data(filepath):
    df = pd.read_csv(filepath)

    # Nettoyage et ajout de features
    df_clean = clean_data(df)
    df_feat = add_features(df_clean)

    # Préparation des données pour le modèle
    X, y = prepare_model_data(df_feat)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardisation
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(X_train_scaled, y_train):
    with mlflow.start_run(run_name="GradientBoostingRegressor") as run:
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
        model.fit(X_train_scaled, y_train)

        # Signature automatique
        signature = infer_signature(X_train_scaled, model.predict(X_train_scaled))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="gboost-ghg-predictor",
            signature=signature,
            input_example=X_train_scaled[:2]
        )

        mlflow.log_params({
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3
        })

    return model


def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print(f"📊 RMSE : {rmse:.4f}")
    print(f"📈 R²   : {r2:.4f}")

    mlflow.log_metric("eval_RMSE", rmse)
    mlflow.log_metric("eval_R2", r2)

    # Visualisation
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='w', s=60)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("True GHG emissions")
    plt.ylabel("Predicted")
    plt.title("Gradient Boosting: Predictions vs Reality")
    plt.grid(True)

    os.makedirs("model", exist_ok=True)
    plot_path = "model/evaluation_plot.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)


def save_model(bundle, path="gb_model.pkl"):
    joblib.dump(bundle, path)
    print(f"✔ Modèle sauvegardé dans : {path}")


def load_model(path="gb_model.pkl"):
    model_bundle = joblib.load(path)
    print("✔ Modèle chargé avec succès.")
    return model_bundle
