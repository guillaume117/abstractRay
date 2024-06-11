
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

RUN pip install ray[default] torch tqdm

COPY . /app

EXPOSE  8265
WORKDIR /app

CMD ["python", "app/main.py", "--num_chanel", "3", "--width", "112", "--height", "112",  "--chunk_size", "1" ,"--num_worker", "10","--alpha", "0.0001"]
