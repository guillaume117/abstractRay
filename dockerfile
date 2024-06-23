
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

RUN pip install ray[default] torch tqdm  torchvision

COPY . /app

EXPOSE  8265
WORKDIR /app

CMD ["python", "app/main.py"]
