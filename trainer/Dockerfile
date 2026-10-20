FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/trainer:/app/simulator
WORKDIR /app
RUN pip install --no-cache-dir "gymnasium==0.29.1" "numpy<2" "Pillow>=10,<11"
COPY trainer /app/trainer
COPY simulator/protocol.py /app/simulator/protocol.py
CMD ["python", "/app/trainer/smoke_test.py", "--steps", "30"]
