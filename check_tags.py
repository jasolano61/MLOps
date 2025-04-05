# ✅ check_tags.py
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("file:///mlruns")
client = MlflowClient()

model_name = "MLOPs_model"
alias = "champion"

try:
    mv = client.get_model_version_by_alias(model_name, alias)
    version = mv.version

    print(f"\n📅 Modelo: {model_name}")
    print(f"🏷️ Alias: {alias} → Versión: {version}\n")
    tags = client.get_model_version(model_name, version).tags
    if tags:
       print("💼 Tags asociados:")
       for k, v in tags.items():
            print(f" - {k}: {v}")
    else:
       print("⚠️ Sin tags registrados.")

    run_id = mv.run_id
    metrics = client.get_run(run_id).data.metrics
    print("\n📊 Métricas del entrenamiento:")
    for k, v in metrics.items():
        print(f" - {k}: {v:.4f}")

except Exception as e:
    print(f"\n❌ Error obteniendo info de '{model_name}@{alias}':\n{e}")
