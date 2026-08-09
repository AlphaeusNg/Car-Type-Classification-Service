# Use Python 3.12 slim base image for better compatibility
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy inference-only requirements first to leverage Docker cache
COPY requirements-api.txt .

# Upgrade pip first, then install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements-api.txt

# Inference needs read-only application/model access, not root privileges.
RUN groupadd --system app && \
    useradd --system --gid app --create-home --home-dir /home/app app

# Copy application code and the selected supported model artifact
ARG MODEL_PATH=best_car_model.keras
COPY --chown=app:app api/ api/
COPY --chown=app:app ${MODEL_PATH} ${MODEL_PATH}
COPY --chown=app:app class_mapping.json .

# Expose port 8000
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV HOME=/home/app

USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
