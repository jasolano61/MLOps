import os
import mlflow
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.tracking import MlflowClient
import joblib

# Directorios
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MODEL_PATH = "model.pkl"


workspace_dir = os.getenv("GITHUB_WORKSPACE", os.getcwd())
MLRUNS_URI = os.path.join(workspace_dir, "mlruns")

os.makedirs(MLRUNS_URI, exist_ok=True)

# Fuerza explícitamente la ruta absoluta para MLflow
mlflow.set_tracking_uri(f"file://{MLRUNS_URI}")
client = MlflowClient(tracking_uri=f"file://{MLRUNS_URI}")

# Crear experimento con ubicación explícita de artefactos
experiment_name = "MLOPs"
experiment = client.get_experiment_by_name(experiment_name)

if experiment is None:
    print(f"✅ Creando nuevo experimento '{experiment_name}' en: {MLRUNS_URI}")
    experiment_id = client.create_experiment(experiment_name, artifact_location=MLRUNS_URI)
else:
    print(f"🔄 Usando experimento '{experiment_name}' ya existente.")
    experiment_id = experiment.experiment_id

mlflow.set_experiment(experiment_name)

print(f"🗃️ MLflow tracking URI configurado en: {MLRUNS_URI}")

# Preparación de datos
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

data = fetch_california_housing(as_frame=True)
df = data.frame
df.to_csv(f"{RAW_DIR}/housing_full.csv", index=False)

# División
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_df = X_train.copy()
train_df["MedHouseVal"] = y_train
train_df.to_csv(f"{PROCESSED_DIR}/train.csv", index=False)

test_df = X_test.copy()
test_df["MedHouseVal"] = y_test
test_df.to_csv(f"{PROCESSED_DIR}/test.csv", index=False)

# Modelo
print("🏷️ Entrenar modelo !!")
model = LinearRegression()
model.fit(X_train, y_train)

print("🏷️ Evaluar modelo !!")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("🏷️ Guardar local !!")
joblib.dump(model, MODEL_PATH)

print("🏷️ Registrar en MLflow !!")
with mlflow.start_run() as run:
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # ✅ Especificar modelo con ruta clara
    mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="MLOPs_model")

    model_uri = f"runs:/{run.info.run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name="MLOPs_model")

    client.set_registered_model_alias("MLOPs_model", "champion", mv.version)
    client.set_model_version_tag("MLOPs_model", mv.version, "validation_status", "passed")
    client.set_model_version_tag("MLOPs_model", mv.version, "author", "JASQ")
    client.set_model_version_tag("MLOPs_model", mv.version, "score", f"{r2:.4f}")

    if r2 > 0.5:
        client.transition_model_version_stage(
            name="MLOPs_model",
            version=mv.version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"🚀 Versión {mv.version} promovida a 'Production'")

    print(f"✅ Modelo registrado como 'MLOPs_model' versión {mv.version}")
    print(f"MSE: {mse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")