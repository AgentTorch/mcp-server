# Use a Python 3.11 base image
FROM python:3.11

# Set working directory
WORKDIR /app

# Install system dependencies including GDAL for spatial libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgdal-dev \
    gdal-bin \
    libproj-dev \
    libspatialindex-dev \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV GDAL_VERSION=3.6.2
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV PROJ_LIB=/usr/share/proj

# Copy requirements and install Python dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install AgentTorch from GitHub
RUN pip install --no-cache-dir git+https://github.com/agenttorch/agenttorch.git

# Create required directories
RUN mkdir -p static templates data/18x25 services

# Copy application code
COPY server.py .
COPY config.yaml .
COPY server.sh .

# Copy pyproject.toml for reference (optional)
COPY pyproject.toml .

# Make the server script executable
RUN chmod +x server.sh

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV AGENTTORCH_CONFIG_PATH=/app/config.yaml

# Expose ports
# WebSocket server port
EXPOSE 8765  
# HTTP server port for UI
EXPOSE 8080

# Entry point script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["docker-entrypoint.sh"]

# Default command runs all components
CMD ["full"]