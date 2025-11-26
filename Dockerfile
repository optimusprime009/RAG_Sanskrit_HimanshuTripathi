# Use a lightweight Python base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install system build tools (required for llama-cpp-python)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY code/requirements.txt .

# Install Python dependencies
# We set CMAKE_ARGS to ensure CPU-only build for llama-cpp
RUN CMAKE_ARGS="-DGGML_BLAS=OFF" pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY code/ ./code/

# Set PYTHONPATH so python can find your modules
ENV PYTHONPATH=/app/code

# Default command (keeps container running for interactive use)
CMD ["python", "code/main.py"]