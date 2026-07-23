# Deployment Guide - Artificial Lift Optimization

## Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python train.py

EXPOSE 5007

CMD ["python", "app.py"]
```

### Build and Run
```bash
docker build -t artificial-lift-optimization .
docker run -p 5007:5007 artificial-lift-optimization
```

### Docker Compose
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "5007:5007"
    environment:
      - FLASK_DEBUG=0
    volumes:
      - ./outputs:/app/outputs
    restart: unless-stopped
```

```bash
docker-compose up -d
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| FLASK_DEBUG | Enable debug mode | 1 |
| PORT | Server port | 5007 |
| HOST | Server host | 0.0.0.0 |

## Manual Deployment

### Prerequisites
- Python 3.8+
- pip

### Steps
```bash
git clone https://github.com/kcabrera83/artificial-lift-optimization.git
cd artificial-lift-optimization
pip install -r requirements.txt
python train.py
python test_api.py  # optional
python app.py
```

## Production Considerations

### Gunicorn (Recommended)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5007 app:app
```

### Security
- Set `DEBUG=False` in production
- Use HTTPS with a reverse proxy
- Add authentication for optimization endpoints
- Validate parameter ranges strictly

### Monitoring
- Monitor `/api/health` for model availability
- Track optimization results and improvements
- Monitor failure prediction distribution
- Log all optimization requests

### Performance
- Optimization endpoint runs 500 iterations by default
- Reduce `n_iterations` for faster response in high-throughput scenarios
- Consider caching optimization results for identical inputs
- Pre-load models at startup

### Operational Safety
- Optimization suggestions should be reviewed by engineers
- Set conservative parameter bounds based on equipment specs
- Monitor failure predictions for early warning systems
- Regular model retraining with field data

## CI/CD
GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push.

## API Self-Documentation
Access OpenAPI docs at: `http://localhost:5007/api/docs`
