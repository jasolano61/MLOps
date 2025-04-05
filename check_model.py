import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("file:///mlruns")
client = MlflowClient()

# Ver versión de MLflow
import mlflow
print(f"📦 MLflow versión: {mlflow.__version__}\n")

# Mostrar modelos registrados
print("📂 Modelos registrados:")
models = client.search_registered_models()
if not models:
    print("❌ No hay modelos registrados.")
else:
    for m in models:
        print(f" - {m.name}")

# Intentar recuperar alias champion
model_name = "MLOPs_model"
try:
    mv = client.get_model_version_by_alias(model_name, "champion")
    print(f"\n🏷️ Alias 'champion' del modelo '{model_name}' apunta a versión {mv.version}")
except Exception as e:
    print(f"\n⚠️ Alias 'champion' no encontrado para el modelo '{model_name}':\n{e}")