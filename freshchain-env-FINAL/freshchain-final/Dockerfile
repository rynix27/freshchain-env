# FreshChain Post-Harvest Yield Loss Environment
# Dockerfile for Hugging Face Spaces (port 7860)

FROM python:3.11-slim

# Required for HF Spaces
LABEL maintainer="FreshChain Team"
LABEL org.opencontainers.image.description="FreshChain OpenEnv - Post-Harvest Yield Loss RL Environment"

WORKDIR /app

# Install dependencies
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY . .

# HF Spaces runs as non-root user
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

# Expose HF Spaces standard port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/health')" || exit 1

# Start the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
