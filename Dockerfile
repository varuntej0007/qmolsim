FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    python3-rdkit \
    librdkit-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code
COPY . .

# Create output directories
RUN mkdir -p reports static/structures data/results phase2/data phase3/data

EXPOSE 5000 5001

CMD ["python", "run_all.py"]
