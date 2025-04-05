from prefect import flow, task
import subprocess
import os

@task
def entrenar_modelo():
    print("🔁 Entrenando modelo...")
    result = subprocess.run(["python", "src/train.py"], capture_output=True, text=True)

    print(result.stdout)
    if result.returncode != 0:
        print("❌ Error al entrenar:")
        print(result.stderr)
        raise Exception("❌ Falló la tarea de entrenamiento")

    print("✅ Modelo entrenado y registrado correctamente.")

@task
def reiniciar_api():
    print("♻️ Reinicio de FastAPI omitido en contenedor Prefect.")
    print("ℹ️ Ejecutá 'make restart-api' desde el host si es necesario.")

@flow(name="Flujo completo de MLOps")
def flujo_mlops():
    entrenar_modelo()
    reiniciar_api()

if __name__ == "__main__":
    flujo_mlops()
