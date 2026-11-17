# !/bin/sh
# Regenerates requirements.lock from requirements.txt in a clean container.
set -e
docker run --rm -v "$PWD":/work -w /work python:3.12-slim \
  sh -c "pip install -q --no-cache-dir -r requirements.txt && pip freeze" > requirements.lock
echo "requirements.lock updated:"
cat requirements.lock