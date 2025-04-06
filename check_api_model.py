import mlflow
from mlflow.tracking import MlflowClient
import os

mlflow.set_tracking_uri("file:///app/mlruns")
client = MlflowClient()

model_name = "MLOPs_model"
alias = "champion"

try:
    mv = client.get_model_version_by_alias(model_name, alias)
    print(f"✅ Alias '{alias}' apunta a versión {mv.version}")
    print(f"📁 Source: {mv.source}")
    
    # Verificar si existe el path del modelo
    if os.path.exists(mv.source):
        print("✅ El modelo existe físicamente en el contenedor.")
    else:
        print("❌ El modelo NO se encuentra en la ruta indicada.")

except Exception as e:
    print(f"❌ Error: {e}")