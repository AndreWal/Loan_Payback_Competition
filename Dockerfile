# Base image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies into the image's Python
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Default command: interactive shell
CMD ["bash"]
