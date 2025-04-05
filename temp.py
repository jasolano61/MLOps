import mlflow

# Crear el cliente MLflow
client = mlflow.tracking.MlflowClient()

# Listar todos los experimentos
experiments = client.list_experiments()

# Imprimir el nombre de los experimentos
for experiment in experiments:
    print(f"Experimento: {experiment.name}")