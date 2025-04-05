FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y git  # Instalacion git

RUN pip install --upgrade pip && pip install -r requirements.txt

RUN python src/train.py

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
