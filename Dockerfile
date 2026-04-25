FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git gcc g++ curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p data results/plots
RUN python scripts/generate_plots.py || echo "Plot generation skipped"
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s \
  CMD curl -f http://localhost:7860/health || exit 1
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
