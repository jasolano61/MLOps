@echo off
docker run --rm ^
  -v %cd%\mlruns:/mlruns ^
  -v %cd%\check_tags.py:/app/check_tags.py ^
  check-model ^
  python check_tags.py