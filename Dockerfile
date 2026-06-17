FROM python:3.12-slim

# System libraries: ffmpeg + libsndfile back the audio deps (librosa/soundfile)
# declared in requirements.txt; build-essential covers any wheels needing compilation.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Install Python dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

EXPOSE 8000

# Entry point matches server.py (FastAPI app object `app`)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
