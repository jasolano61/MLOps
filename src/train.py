import os
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib

# Obtener el directorio de trabajo en GitHub Actions
workspace_dir = os.getenv("GITHUB_WORKSPACE", os.getcwd())

# Definir rutas
RAW_DIR = os.path.join(workspace_dir, "data/raw")
PROCESSED_DIR = os.path.join(workspace_dir, "data/processed")
MODEL_PATH = os.path.join(workspace_dir, "model.pkl")
MLRUNS_URI = os.path.join(workspace_dir, "mlruns")

# Crear directorios necesarios
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MLRUNS_URI, exist_ok=True)

# Variables dinámicas
model_name = os.getenv("MODEL_NAME", "MLOPs_model")
mlflow.set_tracking_uri(f"file://{MLRUNS_URI}")

# Verificar y crear experimento si no existe
client = MlflowClient()
experiment = client.get_experiment_by_name("MLOPs")
if experiment is None:
    experiment_id = client.create_experiment("MLOPs")
    print(f"✨ Experimento 'MLOPs' creado con ID {experiment_id}.")
else:
    experiment_id = experiment.experiment_id
    print(f"🔄 Experimento 'MLOPs' encontrado.")

# Cargar dataset
print("📥 Cargando dataset...")
data = fetch_california_housing(as_frame=True)
df = data.frame
df.to_csv(f"{RAW_DIR}/housing_full.csv", index=False)

# Dividir datos
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Guardar datasets
train_df = X_train.copy()
train_df["MedHouseVal"] = y_train
train_df.to_csv(f"{PROCESSED_DIR}/train.csv", index=False)

# Entrenar modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluar modelo
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Guardar modelo localmente
joblib.dump(model, MODEL_PATH)

# Registrar modelo en MLflow

with mlflow.start_run(experiment_id=experiment_id) as run:
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    print("🏷️ Antes mlflow.sklearn.log_model(model, artifact_path='model')")
    mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=model_name)

    model_uri = f"runs:/{run.info.run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)

    client.set_registered_model_alias(model_name, "champion", mv.version)
    client.set_model_version_tag(model_name, mv.version, "validation_status", "passed")
    client.set_model_version_tag(model_name, mv.version, "author", "JASQ")
    client.set_model_version_tag(model_name, mv.version, "score", f"{r2:.4f}")

    print("🏷️ Antes de if r2 . . .")

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
