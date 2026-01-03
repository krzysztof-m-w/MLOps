FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./src/service.py ./src/service.py
COPY ./meta ./meta
COPY ./model.onnx ./model.onnx
COPY ./model.onnx.data ./model.onnx.data

EXPOSE 3000

WORKDIR /app/src
CMD ["bentoml", "serve"]
