# One-off image: converts moravianai/CzechSpeech to the CTranslate2 format.
# Needs transformers + torch, which the runtime image does not, so it lives here.
FROM python:3.12-slim

# HF_HOME=/tmp/hf: the original (unconverted) weights are downloaded inside the
# container and disappear with it; only the converted model lands in /models.
ENV PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    HF_HOME=/tmp/hf

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# Same faster-whisper version as the runtime image -> same CTranslate2 converter.
RUN pip install --no-cache-dir faster-whisper==1.2.1 transformers

CMD ["ct2-transformers-converter", \
     "--model", "moravianai/CzechSpeech", \
     "--output_dir", "/models/czechspeech-ct2", \
     "--copy_files", "tokenizer.json", "preprocessor_config.json", \
     "--quantization", "int8"]
