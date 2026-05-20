# Use Ubuntu 24.04 LTS as base image
FROM ubuntu:24.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Copy all files
COPY . .

# Make reproduce.sh executable
RUN chmod +x reproduce.sh

# Default command
CMD ["bash", "reproduce.sh"]