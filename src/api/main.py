"""
FastAPI application for DroneRescue model serving.
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time
from datetime import datetime
from pathlib import Path
import yaml
from loguru import logger

from src.api.models import (
    PredictActionRequest,
    PredictActionResponse,
    BatchPredictActionRequest,
    BatchPredictActionResponse,
    HealthResponse,
    ModelInfoResponse,
    ModelsListResponse,
    ErrorResponse,
)
from src.api.service import ModelService


def load_config(config_path: str = "src/api/config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise RuntimeError(f"Configuration loading failed: {e}")


# Global model service instance
model_service: ModelService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    global model_service

    # Startup
    logger.info("Starting DroneRescue API service...")

    # Load configuration
    config = load_config()

    # Initialize model service
    model_service = ModelService(config)

    # Load all configured models
    if not model_service.load_all_models():
        logger.error("Failed to load models during startup")
        raise RuntimeError("Model loading failed")

    loaded_models = model_service.get_loaded_models()
    logger.info(
        f"API service started successfully with {len(loaded_models)} models: {loaded_models}"
    )

    yield

    # Shutdown
    logger.info("Shutting down DroneRescue API service...")


# Create FastAPI app
app = FastAPI(
    title="DroneRescue Model API",
    description="API service for T5-based drone rescue mission action prediction",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_model_service() -> ModelService:
    """Dependency to get model service."""
    if model_service is None:
        raise HTTPException(status_code=503, detail="Model service not available")
    return model_service


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": "DroneRescue Model API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "models": "/models",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(service: ModelService = Depends(get_model_service)):
    """Health check endpoint."""
    loaded_models = service.get_loaded_models()
    total_models = len(service.config["models"])

    return HealthResponse(
        status="healthy" if service.is_any_model_loaded() else "unhealthy",
        models_loaded=loaded_models,
        total_models=total_models,
        uptime_seconds=service.get_uptime(),
    )


@app.get("/models", response_model=ModelsListResponse)
async def list_models(service: ModelService = Depends(get_model_service)):
    """List all configured models and their status."""
    models_info = service.get_model_info()
    return ModelsListResponse(models=models_info)


@app.get("/model/{model_key}/info", response_model=ModelInfoResponse)
async def get_model_info(
    model_key: str, service: ModelService = Depends(get_model_service)
):
    """Get information about a specific model."""
    if not service.is_model_loaded(model_key):
        raise HTTPException(
            status_code=404, detail=f"Model '{model_key}' not found or not loaded"
        )

    info = service.get_model_info(model_key)
    return ModelInfoResponse(**info)


@app.post("/predict", response_model=PredictActionResponse)
async def predict_action(
    request: PredictActionRequest, service: ModelService = Depends(get_model_service)
):
    """
    Predict the next best action for a given mission state.

    This endpoint takes drone rescue mission state and context and generates the next
    best action using the trained T5 model.
    """
    if not service.is_any_model_loaded():
        raise HTTPException(status_code=503, detail="No models loaded")

    # Validate model_key if provided
    if request.model_key and not service.is_model_loaded(request.model_key):
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model_key}' not found or not loaded",
        )

    try:
        result = service.predict_action(
            mission_state=request.mission_state,
            model_key=request.model_key,
            max_length=request.max_length,
            num_beams=request.num_beams,
            temperature=request.temperature,
            repetition_penalty=request.repetition_penalty,
        )

        return PredictActionResponse(**result)

    except Exception as e:
        logger.error(f"Action prediction error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Action prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictActionResponse)
async def batch_predict_actions(
    request: BatchPredictActionRequest,
    service: ModelService = Depends(get_model_service),
):
    """
    Predict next actions for multiple mission states in batch.

    This endpoint processes multiple drone rescue mission states and generates
    next actions for each using the trained T5 model.
    """
    if not service.is_any_model_loaded():
        raise HTTPException(status_code=503, detail="No models loaded")

    # Validate model_key if provided
    if request.model_key and not service.is_model_loaded(request.model_key):
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model_key}' not found or not loaded",
        )

    try:
        result = service.batch_predict_action(
            mission_states=request.mission_states,
            model_key=request.model_key,
            max_length=request.max_length,
            num_beams=request.num_beams,
            temperature=request.temperature,
            repetition_penalty=request.repetition_penalty,
        )

        return BatchPredictActionResponse(**result)

    except Exception as e:
        logger.error(f"Batch action prediction error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Batch action prediction failed: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return ErrorResponse(error="Internal server error", detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    config = load_config()
    server_config = config.get("server", {})
    host = server_config.get("host", "0.0.0.0")
    port = server_config.get("port", 8000)

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
