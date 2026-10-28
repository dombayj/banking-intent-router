# Serves the fine-tuned DistilBERT (seed 44) with confidence routing on CPU.
#   docker build -t banking-intent-router .
#   docker run --rm -p 8000:8000 banking-intent-router   ->  http://localhost:8000
# The model folder is not in git: train it first (see README) or copy it into models/.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
WORKDIR /srv

# CPU-only PyTorch: no CUDA libraries, so the image is ~1 GB smaller.
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

RUN useradd --create-home appuser
ARG MODEL_DIR=models/finetune_lr0.0002_bs32_cw-none_seed44_20261027T171332Z
COPY --chown=appuser ${MODEL_DIR} /srv/model
COPY --chown=appuser app /srv/app
ENV MODEL_DIR=/srv/model
USER appuser
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
