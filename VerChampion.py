from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = "MLOPs_model"

try:
    mv = client.get_model_version_by_alias(model_name, "champion")
    print(f"✅ Alias 'champion' existe: versión {mv.version}")
except Exception as e:
    print(f"❌ Alias no encontrado: {e}")