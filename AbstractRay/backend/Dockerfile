FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


COPY backend/requirements.txt /tmp/

RUN pip install -r /tmp/requirements.txt

WORKDIR /app/AbstractRay/backend
ENV TORCH_HOME="/app/"
ENV PYTHONPATH="/app:${PYTHONPATH}"

RUN mkdir -p /app/AbstractRay/backend/src

# Copiez les fichiers dans la bonne structure
COPY backend/main_stream.py /app/AbstractRay/backend/
COPY backend/src/ /app/AbstractRay/backend/src/
COPY result/ /app/AbstractRay/result/
RUN touch /app/AbstractRay/__init__.py

EXPOSE 8000

CMD ["uvicorn", "main_stream:app", "--host", "0.0.0.0", "--port", "8000"]
