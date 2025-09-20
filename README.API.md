# DroneRescue API Service

A FastAPI-based service for serving T5 models trained for drone rescue mission action prediction. The service supports multiple models loaded from MLFlow and provides comprehensive inference capabilities with metrics logging.

## Features

- **Multiple Model Support**: Load and serve multiple T5 models simultaneously
- **MLFlow Integration**: Load models from MLFlow runs and log inference metrics
- **RESTful API**: Clean REST API with comprehensive documentation
- **Batch Processing**: Support for single and batch inference requests
- **Health Monitoring**: Health checks and model status endpoints
- **Docker Support**: Containerized deployment with Docker
- **Comprehensive Logging**: Structured logging with loguru

## API Endpoints

### Core Endpoints

- `GET /` - API information and available endpoints
- `GET /health` - Health check with model status
- `GET /models` - List all configured models and their status
- `GET /model/{model_key}/info` - Get detailed information about a specific model

### Inference Endpoints

- `POST /predict` - Single mission state prediction
- `POST /predict/batch` - Batch prediction for multiple mission states

## Configuration

The service uses a YAML configuration file (`src/api/config.yaml`) to define:

- Server settings (host, port)
- MLFlow connection details
- Model configurations with MLFlow run IDs
- Default inference parameters

### Example Configuration

```yaml
# Server configuration
server:
  host: "0.0.0.0"
  port: 8000

# MLFlow configuration
mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "drone-rescue-inference"

# Model configuration
models:
  primary:
    name: "primary-model"
    run_id: "your-primary-run-id-here"
    model_name: "t5-small"
    description: "Primary trained model"

  secondary:
    name: "secondary-model"
    run_id: "your-secondary-run-id-here"
    model_name: "t5-small"
    description: "Secondary model for comparison"
```

## Usage

### Running the Service

#### Local Development

```bash
# Install dependencies
poetry install

# Run the service
python -m src.api.main
```

#### Docker Deployment

```bash
# Build and run with docker-compose
docker-compose -f docker-compose.api.yml up --build

# Or build and run manually
docker build -f Dockerfile.api -t drone-rescue-api .
docker run -p 8000:8000 drone-rescue-api
```

### API Usage Examples

#### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "mission_state": "Drone at position (100, 200) with battery 80%. Target at (300, 400).",
    "model_key": "primary",
    "max_length": 150,
    "num_beams": 2,
    "temperature": 1.0,
    "repetition_penalty": 2.5
  }'
```

#### Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "mission_states": [
      "Drone at position (100, 200) with battery 80%.",
      "Drone battery low at 15%. Return to base required."
    ],
    "model_key": "primary"
  }'
```

#### Health Check

```bash
curl "http://localhost:8000/health"
```

## Inference Testing

The service includes a comprehensive inference testing module (`src/api/inference_test.py`) that:

- Sends test samples to the API
- Collects performance metrics
- Logs results to MLFlow with special "inference_test" tags
- Generates detailed CSV reports

### Running Inference Tests

```bash
# Run inference test with config
python src/api/inference_test.py

# Or run programmatically
from src.api.inference_test import send_test_samples_to_api

result = send_test_samples_to_api(
    api_base_url="http://localhost:8000",
    model_key="primary",
    max_samples=20
)
```

## MLFlow Integration

The service integrates with MLFlow for:

- **Model Loading**: Load trained models from MLFlow runs
- **Metrics Logging**: Log inference metrics with special tags
- **Artifact Management**: Store inference results and reports

### MLFlow Tags

- `inference_test: true` - Marks inference test runs
- `test_type: api_inference` - Specifies the type of inference test

## Monitoring and Logging

- **Structured Logging**: Uses loguru for comprehensive logging
- **Health Checks**: Built-in health monitoring endpoints
- **Performance Metrics**: Detailed timing and performance data
- **Error Handling**: Comprehensive error handling and reporting

## Development

### Project Structure

```
src/api/
├── main.py              # FastAPI application
├── models.py            # Pydantic models
├── service.py           # Model service with MLFlow integration
├── inference_test.py    # Inference testing utilities
└── config.yaml          # Service configuration
```

### Dependencies

- FastAPI - Web framework
- PyTorch - Model inference
- Transformers - T5 model support
- MLFlow - Model management and logging
- Loguru - Structured logging
- Pydantic - Data validation

## Deployment

The service is designed for production deployment with:

- Docker containerization
- Health checks
- Graceful startup/shutdown
- Configuration management
- Comprehensive error handling

For production deployment, ensure:

1. MLFlow server is running and accessible
2. Model run IDs are correctly configured
3. Sufficient resources are allocated
4. Monitoring and logging are properly configured
