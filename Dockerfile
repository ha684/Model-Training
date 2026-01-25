# Base image với CUDA support
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda

# Install Python 3.11 và dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY qwen3_vl/ ./qwen3_vl/
COPY pipeline/ ./pipeline/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY reports/ ./reports/
COPY pipeline.py ./
COPY .env.example ./

# Create necessary directories
RUN mkdir -p runs outputs data

# Copy .env if exists (optional - better to mount at runtime)
# COPY .env ./ 

# Set permissions
RUN chmod +x scripts/*.sh

# Expose Prefect server port (nếu chạy server trong container)
EXPOSE 4200

# Default command
CMD ["python", "pipeline.py"]
