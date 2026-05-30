FROM python:3.10-slim

WORKDIR /app

# system dependencies (CRITICAL FIX)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    gfortran \
    libffi-dev \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "model_server.py"]