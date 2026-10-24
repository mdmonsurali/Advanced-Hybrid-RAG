# check=skip=SecretsUsedInArgOrEnv
# (false positive: AUTH_DATA_FILE is a file path, not a secret)
FROM python:3.11.15-slim-trixie

ENV PYTHONUNBUFFERED=1

# upgrade: pick up Debian security fixes newer than the base image.
RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install --no-install-recommends -y curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY auth_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY auth_service/src /app/src

# Users + sessions JSON store; mounted as a named volume by docker-compose.
ENV AUTH_DATA_FILE=/data/auth.json
RUN mkdir -p /data

WORKDIR /app/src

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
