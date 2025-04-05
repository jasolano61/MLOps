# modificar 0.5 por 0.6 min.
import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib

# Asegúrate de tener configurada la URI de MLRUNS
MLRUNS_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///mlruns")  # Usar URI local para ejemplos simples
mlflow.set_tracking_uri(MLRUNS_URI)

# Configurar rutas
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MODEL_PATH = "model.pkl"

# Obtener el directorio de trabajo de GitHub Actions
workspace_dir = os.getenv("GITHUB_WORKSPACE", os.getcwd())

# Establecer la ubicación de MLRUNS en el directorio de trabajo
MLRUNS_URI = os.path.join(workspace_dir, "mlruns")

# Crear el directorio mlruns si no existe
os.makedirs(MLRUNS_URI, exist_ok=True)

# Configurar el URI de MLRUNS
mlflow.set_tracking_uri(MLRUNS_URI)

# Variables dinámicas
model_name = os.getenv("MODEL_NAME", "MLOPs_model")
mlflow.set_tracking_uri(MLRUNS_URI)
mlflow.set_experiment("MLOPs")

# Crear directorios si no existen
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Cargar dataset
data = fetch_california_housing(as_frame=True)
df = data.frame
df.to_csv(f"{RAW_DIR}/housing_full.csv", index=False)

# Dividir datos
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Guardar conjuntos
train_df = X_train.copy()
train_df["MedHouseVal"] = y_train
train_df.to_csv(f"{PROCESSED_DIR}/train.csv", index=False)

test_df = X_test.copy()
test_df["MedHouseVal"] = y_test
test_df.to_csv(f"{PROCESSED_DIR}/test.csv", index=False)

print(f"🏷️ Entrenar modelo !!")

# Entrenar modelo
model = LinearRegression()
model.fit(X_train, y_train)

print(f"🏷️ Evaluar modelo !!")

# Evaluar modelo
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"🏷️ Guardar local !!")

# Guardar local
joblib.dump(model, MODEL_PATH)

print(f"🏷️ Registrar en MLflow !!")

# Registrar en MLflow
client = MlflowClient()

print(f"🏷️ Antes del with !")

with mlflow.start_run() as run:
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    model_uri = f"runs:/{run.info.run_id}/model"

    print(f"🏷️ Antes mlflow.sklearn.log_model(model, artifact_path="model")")

    mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"🏷️ mlflow.register_model: {model_uri} !")

    mv = mlflow.register_model(model_uri=model_uri, name=model_name)

    client.set_registered_model_alias(model_name, "champion", mv.version)

    print(f"🏷️ Alias 'champion' asignado a versión {mv.version}")

    client.set_model_version_tag(model_name, mv.version, "validation_status", "passed")
    client.set_model_version_tag(model_name, mv.version, "author", "JASQ")
    client.set_model_version_tag(model_name, mv.version, "score", f"{r2:.4f}")

    if r2 > 0.5:
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"🚀 Versión {mv.version} promovida a 'Production'")

    print(f"✅ Modelo registrado como '{model_name}' versión {mv.version}")
    print(f"MSE: {mse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
