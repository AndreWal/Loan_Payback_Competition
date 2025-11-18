# Create the Docker yaml file
cat > Docker/docker-compose.yml << 'EOF'
services:
  app:
    image: loanimage:latest          # image you built with `docker build`
    container_name: loan_app
    working_dir: /app
    volumes:
      - .:/app
    command: ["bash"]
EOF

# Create a Dockerfile
cat > Dockerfile << 'EOF'
# Base image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies into the image's Python
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Default command: interactive shell
CMD ["bash"]
EOF