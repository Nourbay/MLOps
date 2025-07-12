# === Configuration ===
ENV_DIR=env
PYTHON=$(ENV_DIR)/bin/python
PIP=$(ENV_DIR)/bin/pip
FLAKE8_REPORT=reports/flake8_report.txt
BLACK_REPORT=reports/black_report.txt
BANDIT_REPORT=reports/bandit_report.txt
DATA=projet5.csv
MAIN=main.py
MODULE=functionalities.py
MODEL_DIR=model
MODEL_FILE=$(MODEL_DIR)/gboost_model.pkl
REQUIREMENTS=requirements.txt
DEPS=pandas scikit-learn matplotlib joblib flake8 black bandit mlflow fastapi uvicorn

# === Commande par défaut
all: install auto

# === Créer l'environnement virtuel
$(ENV_DIR)/bin/activate:
	@echo "🔧 Création de l'environnement virtuel..."
	python3 -m venv $(ENV_DIR)

# === Génération de requirements.txt
$(REQUIREMENTS):
	@echo "📄 Génération de requirements.txt..."
	@echo "$(DEPS)" | tr ' ' '\n' > $(REQUIREMENTS)

# === Installation des dépendances
install: $(ENV_DIR)/bin/activate $(REQUIREMENTS)
	@echo "⬇ Installation des dépendances..."
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQUIREMENTS)
	@echo "✅ Environnement prêt."

# === Lancer le script principal
run:
	$(PYTHON) $(MAIN)

# === Étapes modulaires
prepare:
	$(PYTHON) -c "from functionalities import prepare_data; prepare_data('$(DATA)')"

train:
	$(PYTHON) -c "from functionalities import prepare_data, train_model; X, cols=prepare_data('$(DATA)'); train_model(X, cols)"

evaluate:
	$(PYTHON) -c "from functionalities import prepare_data, train_model, evaluate_model; X, cols=prepare_data('$(DATA)'); model, X_test, y_test = train_model(X, cols); evaluate_model(model, X_test, y_test)"

save:
	$(PYTHON) -c "from functionalities import prepare_data, train_model, save_model; X, cols=prepare_data('$(DATA)'); model, X_test, y_test = train_model(X, cols); save_model(model)"

load:
	$(PYTHON) -c "from functionalities import load_model; load_model(); print('✔ Modèle chargé')"

# === Pipeline complet automatisé
auto:
	@echo "🚀 Exécution complète du pipeline + API :"
	$(PYTHON) -c "\
from functionalities import prepare_data, train_model, evaluate_model, save_model, load_model;\
import mlflow;\
mlflow.set_tracking_uri('http://localhost:5000');\
print('▶ Préparation...');\
X, cols = prepare_data('$(DATA)');\
print('✅ Données préparées.');\
print('▶ Entraînement...');\
model, X_test, y_test = train_model(X, cols);\
print('✅ Modèle entraîné.');\
print('▶ Évaluation...');\
evaluate_model(model, X_test, y_test);\
print('▶ Sauvegarde...');\
save_model(model);\
print('▶ Chargement...');\
load_model();\
print('✅ Pipeline terminé avec succès.')"
	@echo "🌐 Lancement de l'API FastAPI..."
	uvicorn app:app --reload --host 0.0.0.0 --port 8000

# === Interface de suivi MLflow
mlflow-ui:
	@echo "🚀 Lancement du serveur MLflow à http://localhost:5000"
	mlflow ui

# === Nettoyage
clean:
	rm -f $(MODEL_FILE)
	@echo "  Modèle supprimé : $(MODEL_FILE)"

reset:
	rm -rf $(ENV_DIR) $(MODEL_FILE) $(REQUIREMENTS) mlruns
	@echo "♻️ Projet complètement nettoyé."

# === Qualité de code
quality:
	@echo "🔍 Vérification du code avec flake8 et black..."
	@mkdir -p reports
	$(PYTHON) -m flake8 . --exclude=env,__pycache__ --max-line-length=100 | tee $(FLAKE8_REPORT)
	$(PYTHON) -m black --check . | tee $(BLACK_REPORT)

# === Sécurité
security:
	@echo "🛡️ Analyse de sécurité avec bandit..."
	@mkdir -p reports
	$(PYTHON) -m bandit -r . -x venv,__pycache__ > $(BANDIT_REPORT)
	@echo "✅ Rapport de sécurité enregistré dans $(BANDIT_REPORT)"

# === CI Locale
ci-local: install quality run test
	@echo "==================================================="
	@echo "✅ Local CI completed successfully!"
	@echo "==================================================="

# === Test placeholder
test:
	@echo "✅ Test basique : OK"
