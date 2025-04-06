from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import os

mlflow.set_tracking_uri("file:///mlruns")

app = FastAPI()

# 🎯 Permitir CORS en desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Solo para testing local
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📦 Nombre del modelo por entorno
model_name = os.getenv("MODEL_NAME", "MLOPs_model")

# 📥 Estructura esperada en el JSON
class InputData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# 🩺 Endpoint de salud
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/model-info")
def model_info():
    try:
        client = mlflow.tracking.MlflowClient()
        mv = client.get_model_version_by_alias(model_name, "champion")
        tags = client.get_model_version_tags(model_name, mv.version)

        return {
            "model_name": model_name,
            "alias": "champion",
            "version": mv.version,
            "tags": tags
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 🤖 Endpoint de predicción
@app.post("/predict")
def predict(data: InputData):
    try:
        # 🔄 Obtener la versión actual apuntada por alias
        client = MlflowClient()
        mv = client.get_model_version_by_alias(model_name,"champion")
        version = mv.version

        # 🧠 Cargar modelo desde el alias actualizado
        model = mlflow.sklearn.load_model(f"models:/{model_name}@champion")
#        model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")

        # 🧾 Convertir entrada a DataFrame y predecir
        df = pd.DataFrame([data.dict()])
        prediction = model.predict(df)

        return {
            "prediction": prediction.tolist(),
            "model_version": version
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error: {e}")