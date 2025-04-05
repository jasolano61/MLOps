from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd

# Configurar MLflow tracking URI
mlflow.set_tracking_uri("file:///C:/mlops-california-housing/mlruns")

# Usar el último run del experimento (esto se puede automatizar)
run_id = "0b401fa1cbdb4303bf96c2bf6d0c0d62"

# Cargar el modelo desde MLflow
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

# Inicializar FastAPI
app = FastAPI()

# Esquema de entrada usando Pydantic
class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.post("/predict")
def predict(input: HousingInput):
    # Convertir entrada a DataFrame
    data = pd.DataFrame([input.dict()])
    prediction = model.predict(data)
    return {"prediction": prediction[0]}
