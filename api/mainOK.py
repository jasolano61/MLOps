import os
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# 🌐 CORS para desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📦 Cargar modelo por alias
model_name = os.getenv("MODEL_NAME", "MLOPs_model")
try:
    model = mlflow.sklearn.load_model(f"models:/{model_name}@champion")
    print(f"✅ Modelo '{model_name}@champion' cargado correctamente")
#    model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")
#    print(f"✅ Modelo '{model_name}' cargado correctamente")
except Exception as e:
    print(f"❌ Error al cargar el modelo '{model_name}@champion': {e}")
#    print(f"❌ Error al cargar el modelo '{model_name}/latest': {e}")
    model = None

# 🧾 Estructura de entrada
class InputData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputData):
    try:
        model_name = os.getenv("MODEL_NAME", "MLOPs_model")
        client = mlflow.tracking.MlflowClient()
        mv = client.get_model_version_by_alias(model_name, "champion")
        version = mv.version
        model = mlflow.sklearn.load_model(f"models:/{model_name}@champion")

        df = pd.DataFrame([data.dict()])
        prediction = model.predict(df)

        return {
            "prediction": prediction.tolist(),
            "model_version": version
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

