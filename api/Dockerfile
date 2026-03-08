# Use explicit, minimal base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

#install dependensi strictly dari lockfile export
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

#copy code application dan serialized artifacts yang diperlukan 
#(in a true enterprise system, artifacts are pulled from an S3/GCS model registry at build time)
COPY api/ ./api/
COPY src/features/ ./src/features/
COPY data/processed/preprocessor.joblib ./data/processed/
COPY src/models/lgbm_baseline.txt ./src/models/

#expose strict port mapping
EXPOSE 8000

#Execute ASGI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]