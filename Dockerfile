FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data results/plots

# Download datasets at build time (falls back to synthetic on network failure)
RUN python scripts/download_tasks.py --quiet || echo "Dataset download failed — synthetic tasks will be used"

# Pre-generate all plots so Gradio loads instantly
RUN python scripts/generate_plots.py

EXPOSE 8000
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port 8000 & python ui/app.py & wait"]
