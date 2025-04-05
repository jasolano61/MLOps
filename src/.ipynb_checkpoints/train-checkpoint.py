
import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("file:///C:/mlops-california-housing/mlruns")
mlflow.set_experiment("california_housing")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Cargar los datos procesados
train_path = "../data/processed/train.csv"
df = pd.read_csv(train_path)

# Separar variables
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Entrenar modelo
model = LinearRegression()
model.fit(X, y)

# Predicciones
predictions = model.predict(X)

# Métricas
mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

# Iniciar experimentación con MLflow
with mlflow.start_run():
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)
    mlflow.sklearn.log_model(model, "model")

    print(f"✅ Modelo registrado con MSE: {mse:.4f}, R2: {r2:.4f}")
