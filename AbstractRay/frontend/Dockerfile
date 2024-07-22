FROM python:3.11

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN pip install utility


COPY app.py /app/


EXPOSE 8501
WORKDIR /app

CMD ["streamlit", "run", "app.py", "--server.enableWebsocketCompression=false", "--server.enableXsrfProtection=false", " --server.enableCORS=false"]