FROM python:3.9-slim

WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- THIS WAS MISSING: Copy all application source code ---
COPY . .

# Make start script executable
RUN chmod +x start.sh

# Run the startup script
CMD ["./start.sh"]