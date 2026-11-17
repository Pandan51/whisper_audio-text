# Runtime image: only what transcriber.py needs. No torch, no transformers.
FROM python:3.12-slim

# HOME=/tmp: the container may run as your user ID, which has no home folder inside.
# HF_HOME: where the turbo model is downloaded (./models on the server).
# CZECHSPEECH_MODEL_DIR: where the converted CzechSpeech model is read from.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    HF_HOME=/models/hf \
    CZECHSPEECH_MODEL_DIR=/models/czechspeech-ct2

WORKDIR /app

# PyAV wheels bundle their own FFmpeg, so no apt-get install ffmpeg is needed.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY transcriber.py .

# Everything after the service name in `docker compose run` becomes transcriber.py arguments.
ENTRYPOINT ["python", "transcriber.py"]
