# CSLR Backend - Quick Start Guide

## 🚀 Fast Track (5 Minutes)

### 1. Setup
```bash
cd /home/kathir/CSLR/application/backend

# Run automated setup
./setup.sh

# Activate environment
source venv/bin/activate
```

### 2. Start Server
```bash
# Development mode (auto-reload)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode (with uvloop)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4 --loop uvloop
```

### 3. Test API
```bash
# Health check
curl http://localhost:8000/api/v1/health

# System info
curl http://localhost:8000/api/v1/health/system

# Model status
curl http://localhost:8000/api/v1/health/models
```

## 📡 API Usage Examples

### Video Inference
```bash
curl -X POST http://localhost:8000/api/v1/inference/video \
  -F "file=@/path/to/video.mp4" \
  -H "X-API-Key: your_api_key_here"  # if auth enabled
```

### Frame Inference
```python
import requests
import base64
import cv2

# Encode frames
frames = []
for i in range(10):
    frame = cv2.imread(f"frame_{i}.jpg")
    _, buffer = cv2.imencode('.jpg', frame)
    b64 = base64.b64encode(buffer).decode('utf-8')
    frames.append(b64)

# Send request
response = requests.post(
    "http://localhost:8000/api/v1/inference/frames",
    json={"frames": frames, "fps": 25.0}
)

result = response.json()
print(f"Sentence: {result['sentence']}")
print(f"Gloss: {result['gloss']}")
print(f"Confidence: {result['confidence']}")
```

### WebSocket Streaming
```python
import asyncio
import websockets
import json
import base64
import cv2

async def stream_video():
    uri = "ws://localhost:8000/api/v1/ws/inference"
    
    async with websockets.connect(uri) as websocket:
        cap = cv2.VideoCapture(0)  # Webcam
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', frame)
            b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send frame
            await websocket.send(json.dumps({
                "frame": b64,
                "timestamp": time.time()
            }))
            
            # Receive result
            response = await websocket.recv()
            result = json.loads(response)
            
            print(f"Live: {result['sentence']}")
            
            await asyncio.sleep(0.033)  # ~30 FPS

asyncio.run(stream_video())
```

## 🔐 Security Setup

### Generate API Key
```bash
python3 scripts/generate_api_key.py
```

### Enable Authentication
Edit `.env`:
```bash
REQUIRE_AUTH=true
API_KEYS='your_generated_api_key_here'
```

### Use API Key
```bash
curl -H "X-API-Key: your_api_key" http://localhost:8000/api/v1/health
```

## 🐳 Docker Usage

### Build
```bash
docker build -t cslr-backend:latest .
```

### Run with GPU
```bash
docker run --gpus all \
  -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/data:/app/data \
  --env-file .env \
  cslr-backend:latest
```

### Check GPU in Container
```bash
docker exec <container_id> nvidia-smi
```

## 🧪 Testing

### Run All Tests
```bash
# Full test suite
pytest

# With coverage
pytest --cov=app --cov-report=html --cov-report=term-missing

# Verbose output
pytest -v
```

### Test Categories
```bash
# API integration tests
python3 tests/test_api_integration.py

# Unit tests
pytest tests/test_models.py
pytest tests/test_preprocessing.py
pytest tests/test_security.py

# Specific test
pytest tests/test_health.py::test_health_check
```

### Integration Testing
```bash
# API tests (requires running server)
python3 tests/test_api_integration.py

# With video file
python3 tests/test_api_integration.py /path/to/test.mp4

# Health check
./health_check.sh
```

## 📊 Monitoring

### Check Logs
```bash
# Tail logs
tail -f logs/app.log

# Filter errors
grep ERROR logs/app.log
```

### GPU Monitoring
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Check CUDA availability
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### API Documentation
Visit http://localhost:8000/docs for interactive Swagger UI

## 🛠 Development

### Install Dev Dependencies
```bash
pip install pytest pytest-cov black isort mypy
```

### Code Quality
```bash
# Format code
black app/
isort app/

# Type check
mypy app/

# Run tests
pytest tests/ -v
```

### Export Models
```bash
# ONNX export
python3 scripts/export_onnx.py --checkpoint checkpoints/best.pth

# TorchScript export
python3 scripts/export_torchscript.py --checkpoint checkpoints/best.pth
```

## 🔧 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
export BATCH_SIZE=1

# Enable AMP
export USE_AMP=true

# Clear cache
python3 -c "import torch; torch.cuda.empty_cache()"
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Verify installation
python3 -c "import torch, cv2, mediapipe; print('OK')"
```

### Slow Inference
```bash
# Check GPU usage
nvidia-smi

# Enable cuDNN benchmark
export CUDNN_BENCHMARK=true

# Reduce clip length
export CLIP_LENGTH=32
```

## 📚 Documentation

- **Full README**: [README.md](README.md)
- **API Docs**: http://localhost:8000/docs
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)
- **Architecture**: [README.md#architecture](README.md#architecture)

## 🎯 Common Tasks

### Add New Checkpoint
```bash
# Place in checkpoints/ directory
cp /path/to/model.pth checkpoints/rgb_best.pth

# Restart server
pkill -f uvicorn
uvicorn app.main:app --reload
```

### Update Configuration
```bash
# Edit .env
nano .env

# Restart required
```

### Train Model
```bash
curl -X POST http://localhost:8000/api/v1/training/start \
  -H "Content-Type: application/json" \
  -d '{
    "data_dir": "/data/wlasl",
    "num_epochs": 100,
    "batch_size": 4,
    "learning_rate": 0.001
  }'
```

## 🚨 Emergency Commands

### Kill All Processes
```bash
pkill -f uvicorn
pkill -f python3
```

### Reset Environment
```bash
rm -rf venv/
./setup.sh
```

### Clear All Logs
```bash
rm -rf logs/*.log
```

---

**Need Help?** Check [README.md](README.md) or [CHANGELOG.md](CHANGELOG.md)
