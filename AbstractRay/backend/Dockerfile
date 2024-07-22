FROM python:3.11


COPY requirements.txt /tmp/

RUN pip install -r /tmp/requirements.txt

WORKDIR /app
ENV TORCH_HOME="/app/"

COPY main_stream_2.py /app/ 
COPY src/ ./src/



EXPOSE 8000

CMD ["uvicorn", "main_stream_2:app", "--host", "0.0.0.0", "--port", "8000"]
