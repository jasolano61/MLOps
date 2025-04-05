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

