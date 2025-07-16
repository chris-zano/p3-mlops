FROM python:3.10

WORKDIR /app

COPY scripts/train.py .

COPY requirements.txt .

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "scripts/train.py"]
