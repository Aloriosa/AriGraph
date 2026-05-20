FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3 as default
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /workspace

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy all other files
COPY . .

# Make reproduce.sh executable
RUN chmod +x reproduce.sh

# Default command
CMD ["bash", "reproduce.sh"]