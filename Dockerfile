FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libxml2-dev \
    libxslt-dev \
    zlib1g-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY lm-polygraph ./lm-polygraph

RUN pip install --upgrade pip && pip install -e ./lm-polygraph

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .