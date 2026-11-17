# transcriber-service

Audio file in, transcript out. `transcriber.py` in Docker, two models:
`turbo` (any language) and `czechspeech` (Czech, converted once).

## Layout

```
transcriber-service/
  transcriber.py        the code
  requirements.txt      pinned runtime dependencies
  Dockerfile            runtime image
  convert/Dockerfile    one-off image that converts CzechSpeech
  docker-compose.yml    both services + folders mounted into them
  models/               filled on the server (turbo cache, czechspeech-ct2)
  data/input/           put audio here
  data/output/          results appear here
```

## On the server (no sudo)

```bash
cd ~/transcriber-service

# 1. Run containers as your own user (skip with rootless Docker)
echo "HOST_UID=$(id -u)" > .env
echo "HOST_GID=$(id -g)" >> .env

# 2. Build both images
docker compose build
docker compose --profile setup build convert

# 3. Convert CzechSpeech once (result: models/czechspeech-ct2/)
docker compose run --rm convert

# 4. Transcribe
docker compose run --rm transcriber /data/input/input.m4a --out /data/output/input.json

# Long recording, survives SSH disconnect:
docker compose run -d --name job1 transcriber /data/input/input.m4a --out /data/output/input.json
docker logs -f job1      # Ctrl+C stops watching, not the job
docker rm job1           # when finished
```

Options: `--model auto|turbo|czechspeech`, `--language cs`, `--out file.txt|file.json`, `--batch-size 4`.
