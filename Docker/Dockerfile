FROM python:3.10

WORKDIR /app

COPY dist/inference_api .

ENTRYPOINT ["python", "scripts/train.py"]
