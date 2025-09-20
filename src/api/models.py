"""
Pydantic models for API request/response schemas.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class PredictActionRequest(BaseModel):
    """Request model for next action prediction."""

    mission_state: str = Field(
        ...,
        description="Current mission state and context",
        min_length=1,
        max_length=10000,
    )
    model_key: Optional[str] = Field(
        None, description="Key of the model to use for prediction"
    )
    max_length: Optional[int] = Field(
        150, description="Maximum length of generated action", ge=10, le=500
    )
    num_beams: Optional[int] = Field(
        2, description="Number of beams for beam search", ge=1, le=10
    )
    temperature: Optional[float] = Field(
        1.0, description="Sampling temperature", ge=0.1, le=2.0
    )
    repetition_penalty: Optional[float] = Field(
        2.5, description="Repetition penalty", ge=1.0, le=5.0
    )


class PredictActionResponse(BaseModel):
    """Response model for next action prediction."""

    predicted_action: str = Field(..., description="Predicted next action")
    input_length: int = Field(..., description="Length of input mission state")
    output_length: int = Field(..., description="Length of predicted action")
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )
    model_key: str = Field(..., description="Key of the model used")
    model_version: str = Field(..., description="Version of the model used")
    run_id: str = Field(..., description="MLFlow run ID of the model")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Response timestamp"
    )


class BatchPredictActionRequest(BaseModel):
    """Request model for batch action prediction."""

    mission_states: List[str] = Field(
        ...,
        description="List of mission states to predict actions for",
        min_items=1,
        max_items=10,
    )
    model_key: Optional[str] = Field(
        None, description="Key of the model to use for prediction"
    )
    max_length: Optional[int] = Field(
        150, description="Maximum length of generated actions", ge=10, le=500
    )
    num_beams: Optional[int] = Field(
        2, description="Number of beams for beam search", ge=1, le=10
    )
    temperature: Optional[float] = Field(
        1.0, description="Sampling temperature", ge=0.1, le=2.0
    )
    repetition_penalty: Optional[float] = Field(
        2.5, description="Repetition penalty", ge=1.0, le=5.0
    )


class BatchPredictActionResponse(BaseModel):
    """Response model for batch action prediction."""

    predicted_actions: List[str] = Field(..., description="Predicted next actions")
    processing_time_ms: float = Field(
        ..., description="Total processing time in milliseconds"
    )
    model_key: str = Field(..., description="Key of the model used")
    model_version: str = Field(..., description="Version of the model used")
    run_id: str = Field(..., description="MLFlow run ID of the model")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Response timestamp"
    )


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Service status")
    models_loaded: List[str] = Field(..., description="List of loaded model keys")
    total_models: int = Field(..., description="Total number of configured models")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Response timestamp"
    )


class ModelInfoResponse(BaseModel):
    """Model information response."""

    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    model_path: str = Field(..., description="Path to model files")
    tokenizer_name: str = Field(..., description="Tokenizer name")
    max_length: int = Field(..., description="Maximum sequence length")
    device: str = Field(..., description="Device used for inference")
    loaded_at: datetime = Field(..., description="When model was loaded")


class ModelsListResponse(BaseModel):
    """Response model for listing all models."""

    models: Dict[str, Dict[str, Any]] = Field(
        ..., description="Dictionary of all configured models with their info"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Response timestamp"
    )


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Error timestamp"
    )
