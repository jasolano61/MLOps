@echo off
docker run -it --rm ^
  -v %cd%\mlruns:/mlruns ^
  -v %cd%:/app ^
  -w /app ^
  python:3.10-slim bash