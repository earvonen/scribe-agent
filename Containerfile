FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml /app/
COPY src /app/src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir . \
    && chgrp -R 0 /app \
    && chmod -R g=u /app

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "scribe_agent"]
