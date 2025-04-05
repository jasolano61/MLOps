from mlflow.tracking import MlflowClient
import os

# Nombre del modelo desde variable de entorno o por defecto
model_name = os.getenv("MODEL_NAME", "MLOPs_model")

client = MlflowClient()

print(f"🔍 Verificando alias 'champion' para el modelo '{model_name}'...")

try:
    mv = client.get_model_version_by_alias(model_name, "champion")
    print(f"✅ Alias 'champion' existe y apunta a la versión {mv.version}")
    print(f"📦 Status: {mv.status} | Stage: {mv.current_stage}")
    print(f"🏷️ Tags:")
    for tag_key, tag_value in mv.tags.items():
        print(f"   - {tag_key}: {tag_value}")
except Exception as e:
    print(f"❌ Alias 'champion' no encontrado o modelo no registrado.")
    print(f"   Error: {e}")
