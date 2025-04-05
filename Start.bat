@echo off
title 🚀 Lanzador MLOps - by JASQ

cd /d C:\mlops

echo Verificando que Docker esté corriendo...
docker info >nul 2>&1
IF ERRORLEVEL 1 (
    echo ❌ Docker no está activo. Por favor, inícialo primero.
    pause
    exit /b
)

echo 🔁 [1/4] Levantando servicios con Docker...
call make up

echo 🚀 [2/4] Ejecutando flujo completo con Prefect...
call make run-flow

echo 🔄 [3/4] Reiniciando FastAPI para cargar el nuevo modelo...
call make restart-api

echo 🌐 [4/4] Abriendo Swagger UI en el navegador...
start http://localhost:8000/docs

echo 📈 [5/5] Abriendo MLflow UI en el navegador...
start http://localhost:5000

echo ✅ Todo corriendo correctamente. Listo para predecir
pause