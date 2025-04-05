@echo off
docker run --rm ^
  -v %cd%\mlruns:/mlruns ^
  -v %cd%:/app ^
  -w /app ^
  python:3.10-slim ^
  sh -c "pip install mlflow pandas scikit-learn && python check_model.py"