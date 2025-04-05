# GitHub commit
auto-commit:
	python auto_commit.py

# 🚀 Levantar todos los servicios (API, UI, etc.)
up:
	docker-compose up --build -d

# 🛑 Detener todos los servicios
down:
	docker-compose down

# 🔁 Reentrenar el modelo manualmente (train.py)
retrain:
	docker-compose run --rm retrain

# ⚙️ Ejecutar el flujo Prefect (pipeline completo)
run-flow:
	docker-compose run --rm prefect-runner

# 🔄 Reiniciar solo FastAPI
restart-api:
	docker restart mlops-api

# ♻️ Reconstruir imagen de FastAPI y reiniciar
refresh-api:
	docker-compose build mlops-api
	docker restart mlops-api

# 📜 Ver logs del contenedor FastAPI
logs-api:
	docker logs -f mlops-api

# 📜 Ver logs del flujo Prefect
logs-flow:
	docker logs -f prefect-runner

# 🔍 Abrir MLflow UI (solo Windows)
open-ui:
	start http://localhost:5000

# 🧹 Limpieza de datasets y modelo local
clean:
	rm -rf data/processed/*.csv model.pkl

# 💣 Limpieza total (incluye experimentos)
clean-hard:
	rm -rf data/processed/*.csv model.pkl mlruns/

