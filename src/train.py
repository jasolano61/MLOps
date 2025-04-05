
import os
import mlflow
from mlflow.tracking import MlflowClient

workspace_dir = os.getenv("GITHUB_WORKSPACE", os.getcwd())
MLRUNS_URI = os.path.join(workspace_dir, "mlruns")

os.makedirs(MLRUNS_URI, exist_ok=True)
mlflow.set_tracking_uri(f"file://{MLRUNS_URI}")
client = MlflowClient(tracking_uri=f"file://{MLRUNS_URI}")

experiment_name = "MLOPs"
experiment = client.get_experiment_by_name(experiment_name)

if experiment is None:
    experiment_id = client.create_experiment(experiment_name, artifact_location=MLRUNS_URI)
else:
    experiment_id = experiment.experiment_id

mlflow.set_experiment(experiment_name)

with mlflow.start_run():
    # Entrenar modelo
    model.fit(X_train, y_train)

    # Evaluar modelo
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Guardar modelo localmente
    joblib.dump(model, MODEL_PATH)

    # Registrar métricas
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # Log modelo SIN usar registered_model_name
    mlflow.sklearn.log_model(model, artifact_path="model")

    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"

    # Crear o verificar modelo registrado
    try:
        client.create_registered_model("MLOPs_model")
    except mlflow.exceptions.MlflowException:
        pass

    # Crear versión del modelo
    mv = client.create_model_version("MLOPs_model", source=model_uri, run_id=run_id)

    # Asignar alias y tags
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