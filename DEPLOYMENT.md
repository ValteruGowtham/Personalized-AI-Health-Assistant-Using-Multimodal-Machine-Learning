# 📦 Deployment Guide

This guide covers deploying the Personalized AI Health Assistant to production environments.

## 🐳 Docker Deployment

### Create Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data models outputs

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
# Build image
docker build -t health-assistant:latest .

# Run container
docker run -p 8501:8501 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models health-assistant:latest
```

---

## ☁️ Cloud Deployment

### AWS Deployment (EC2 + Docker)

1. **Launch EC2 Instance**
   - Instance type: t3.medium (or larger for GPU)
   - AMI: Ubuntu 22.04 LTS
   - Storage: 30GB+

2. **Setup Instance**
   ```bash
   # SSH into instance
   ssh -i your-key.pem ubuntu@your-instance-ip
   
   # Install Docker
   sudo apt update
   sudo apt install -y docker.io docker-compose
   sudo usermod -aG docker ubuntu
   
   # Clone repository
   git clone https://github.com/ValteruGowtham/Personalized-AI-Health-Assistant-Using-Multimodal-Machine-Learning.git
   cd Personalized-AI-Health-Assistant-Using-Multimodal-Machine-Learning
   
   # Build and run
   docker build -t health-assistant .
   docker run -d -p 80:8501 --name health-app health-assistant
   ```

3. **Configure Security Group**
   - Allow inbound traffic on port 80 (HTTP)
   - Allow inbound traffic on port 443 (HTTPS) if using SSL

### Google Cloud Platform (Cloud Run)

1. **Prepare for Cloud Run**
   
   Create `cloudbuild.yaml`:
   ```yaml
   steps:
     - name: 'gcr.io/cloud-builders/docker'
       args: ['build', '-t', 'gcr.io/$PROJECT_ID/health-assistant', '.']
     - name: 'gcr.io/cloud-builders/docker'
       args: ['push', 'gcr.io/$PROJECT_ID/health-assistant']
   images:
     - 'gcr.io/$PROJECT_ID/health-assistant'
   ```

2. **Deploy**
   ```bash
   # Build and push
   gcloud builds submit --config cloudbuild.yaml
   
   # Deploy to Cloud Run
   gcloud run deploy health-assistant \
     --image gcr.io/$PROJECT_ID/health-assistant \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 4Gi \
     --cpu 2
   ```

### Azure (Container Instances)

```bash
# Create resource group
az group create --name health-assistant-rg --location eastus

# Create container registry
az acr create --resource-group health-assistant-rg --name healthassistantacr --sku Basic

# Build and push image
az acr build --registry healthassistantacr --image health-assistant:latest .

# Deploy container
az container create \
  --resource-group health-assistant-rg \
  --name health-assistant \
  --image healthassistantacr.azurecr.io/health-assistant:latest \
  --cpu 2 \
  --memory 4 \
  --registry-login-server healthassistantacr.azurecr.io \
  --ip-address Public \
  --ports 8501
```

---

## 🔒 Security Considerations

### 1. Environment Variables

Create `.env` file (never commit to git):
```bash
# API Keys
OPENAI_API_KEY=your_key_here

# Database
DATABASE_URL=postgresql://user:pass@host:5432/db

# Security
SECRET_KEY=your_secret_key
ALLOWED_HOSTS=your-domain.com

# Model paths
MODEL_PATH=/app/models/final_model.keras
```

Load in Python:
```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
```

### 2. HTTPS/SSL

Use Let's Encrypt for free SSL:
```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo certbot renew --dry-run
```

### 3. Authentication

Add Streamlit authentication:
```python
import streamlit as st
import hmac

def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input("Password", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False

if not check_password():
    st.stop()
```

### 4. Data Privacy (HIPAA Compliance)

For healthcare data:
- ✅ Encrypt data at rest and in transit
- ✅ Implement access controls
- ✅ Maintain audit logs
- ✅ Use secure cloud providers (AWS HIPAA, Azure Healthcare)
- ✅ Sign Business Associate Agreements (BAAs)

---

## 📊 Monitoring & Logging

### Application Monitoring

```python
# Add to app.py
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    filename=f'logs/app_{datetime.now().strftime("%Y%m%d")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Log predictions
logger.info(f"Prediction made: risk={prediction}, patient_id={patient_id}")
```

### Health Checks

```python
# healthcheck.py
import requests
import sys

try:
    response = requests.get('http://localhost:8501/_stcore/health', timeout=5)
    if response.status_code == 200:
        sys.exit(0)
    else:
        sys.exit(1)
except:
    sys.exit(1)
```

### Performance Monitoring

Use Prometheus + Grafana:

```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8501:8501"
  
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

---

## 🚀 CI/CD Pipeline

### GitHub Actions

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy Health Assistant

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.10
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest test_health_assistant.py -v

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to Cloud Run
        uses: google-github-actions/deploy-cloudrun@v0
        with:
          service: health-assistant
          image: gcr.io/${{ secrets.GCP_PROJECT_ID }}/health-assistant
          credentials: ${{ secrets.GCP_SA_KEY }}
```

---

## 🔄 Model Updates

### Versioning Strategy

```python
# models/
# ├── v1.0/
# │   ├── model.keras
# │   └── metadata.json
# ├── v1.1/
# │   ├── model.keras
# │   └── metadata.json
# └── latest -> v1.1/

import os
import shutil

def deploy_new_model(model_path, version):
    """Deploy new model version."""
    version_dir = f"models/{version}"
    os.makedirs(version_dir, exist_ok=True)
    
    # Copy model
    shutil.copy(model_path, f"{version_dir}/model.keras")
    
    # Update symlink
    latest_link = "models/latest"
    if os.path.islink(latest_link):
        os.remove(latest_link)
    os.symlink(version_dir, latest_link)
```

### A/B Testing

```python
import random

def get_model_version(user_id):
    """Route users to different model versions."""
    if hash(user_id) % 100 < 10:  # 10% to new model
        return "v1.1"
    return "v1.0"
```

---

## 📈 Scaling

### Horizontal Scaling

```yaml
# kubernetes-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: health-assistant
spec:
  replicas: 3
  selector:
    matchLabels:
      app: health-assistant
  template:
    metadata:
      labels:
        app: health-assistant
    spec:
      containers:
      - name: health-assistant
        image: gcr.io/project/health-assistant:latest
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
---
apiVersion: v1
kind: Service
metadata:
  name: health-assistant-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8501
  selector:
    app: health-assistant
```

### Load Balancing

```nginx
# nginx.conf
upstream health_assistant {
    least_conn;
    server app1:8501;
    server app2:8501;
    server app3:8501;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://health_assistant;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 💰 Cost Optimization

### Tips for Reducing Costs

1. **Use Spot Instances**: 70% cheaper on AWS/GCP
2. **Auto-scaling**: Scale down during low traffic
3. **Model Optimization**: Use quantization/pruning
4. **Caching**: Cache BERT embeddings for common queries
5. **CDN**: Use CloudFlare for static assets

### Example Auto-scaling (AWS)

```bash
# Create auto-scaling group
aws autoscaling create-auto-scaling-group \
  --auto-scaling-group-name health-assistant-asg \
  --min-size 1 \
  --max-size 5 \
  --desired-capacity 2 \
  --target-group-arns arn:aws:elasticloadbalancing:... \
  --health-check-type ELB \
  --health-check-grace-period 300
```

---

## ✅ Pre-Deployment Checklist

- [ ] All tests passing
- [ ] Environment variables configured
- [ ] SSL certificate installed
- [ ] Database backups configured
- [ ] Monitoring and logging setup
- [ ] Security audit completed
- [ ] Load testing performed
- [ ] Documentation updated
- [ ] Rollback plan prepared
- [ ] Team notified

---

<div align="center">
  <strong>Ready for Production! 🚀</strong>
</div>
