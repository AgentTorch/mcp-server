# Use a Python 3.11 base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including GDAL
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgdal-dev \
    gdal-bin \
    libproj-dev \
    libspatialindex-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV GDAL_VERSION=3.6.2
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV PROJ_LIB=/usr/share/proj

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server.py .
COPY config.yaml .

# Create directories needed for application
RUN mkdir -p data/18x25
RUN mkdir -p services

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV AGENTTORCH_CONFIG_PATH=/app/config.yaml

# Expose ports
# WebSocket server port
EXPOSE 8765  
# HTTP server port for UI (if needed)
EXPOSE 8080

# Command to run both the MCP server and a simple HTTP server for the UI
CMD ["sh", "-c", "python -m http.server 8080 --directory /app/ui & mcp run server.py"]