"""
Model service for loading and managing T5 models with MLFlow integration.
"""
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from pathlib import Path
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
import mlflow
import mlflow.pytorch
from loguru import logger


class ModelService:
    """Service for managing multiple T5 models with MLFlow integration."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model service with configuration.

        Args:
            config: Configuration dictionary containing model settings
        """
        self.config = config
        self.models: Dict[str, Dict[str, Any]] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_time = time.time()

        # Initialize MLFlow
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])

        # Set MLFlow artifacts root to local directory
        import os

        os.environ["MLFLOW_ARTIFACTS_ROOT"] = str(Path.cwd() / "mlartifacts")

        logger.info(f"ModelService initialized with device: {self.device}")
        logger.info(f"MLFlow tracking URI: {config['mlflow']['tracking_uri']}")

    def load_model(self, model_key: str) -> bool:
        """
        Load a specific model by key.

        Args:
            model_key: Key identifying the model in config

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if model_key not in self.config["models"]:
                logger.error(f"Model key '{model_key}' not found in configuration")
                return False

            model_config = self.config["models"][model_key]
            logger.info(
                f"Loading model '{model_key}' from MLFlow run: {model_config['run_id']}"
            )

            # Load model and tokenizer from MLFlow artifacts
            try:
                logger.info(
                    f"Attempting to load model and tokenizer from MLFlow artifacts for run: {model_config['run_id']}"
                )

                # Search recursively for the run ID in mlartifacts
                # Structure: ./mlartifacts/{experiment_id}/{run_id}/artifacts/model_and_tokenizer/
                artifacts_base = Path("mlartifacts")

                if not artifacts_base.exists():
                    raise Exception("mlartifacts directory not found")

                model_found = False
                model_path = None

                # Recursive search for the run ID
                for exp_dir in artifacts_base.iterdir():
                    if exp_dir.is_dir():
                        # Look for the specific run ID directory
                        run_dir = exp_dir / model_config["run_id"]
                        if run_dir.exists():
                            # Look for model_and_tokenizer directory
                            model_tokenizer_dir = (
                                run_dir / "artifacts" / "model_and_tokenizer"
                            )
                            if model_tokenizer_dir.exists():
                                # Check if it contains the required model files
                                required_files = ["config.json", "model.safetensors"]
                                if all(
                                    (model_tokenizer_dir / file).exists()
                                    for file in required_files
                                ):
                                    model_path = model_tokenizer_dir
                                    model_found = True
                                    break

                if model_found:
                    logger.info(f"Found model and tokenizer directory at: {model_path}")
                    # Load model and tokenizer using the same approach as test.py
                    model = T5ForConditionalGeneration.from_pretrained(str(model_path))
                    tokenizer = T5Tokenizer.from_pretrained(str(model_path))
                    model_version = f"mlartifacts-{model_config['run_id']}"
                    logger.info(
                        f"Successfully loaded model and tokenizer from mlartifacts: {model_path}"
                    )
                else:
                    raise Exception(
                        f"Model directory not found for run {model_config['run_id']}"
                    )

            except Exception as e:
                logger.warning(
                    f"Failed to load from mlartifacts {model_config['run_id']}: {e}"
                )
                logger.info("Attempting to load from local model path...")

                # Try to load from local model path
                local_model_path = Path("models/trained_model")
                if (
                    local_model_path.exists()
                    and (local_model_path / "config.json").exists()
                ):
                    model = T5ForConditionalGeneration.from_pretrained(
                        str(local_model_path)
                    )
                    tokenizer = T5Tokenizer.from_pretrained(str(local_model_path))
                    model_version = f"local-{local_model_path.name}"
                    logger.info(
                        f"Loaded model and tokenizer from local path: {local_model_path}"
                    )
                else:
                    # Fallback to pretrained model
                    model = T5ForConditionalGeneration.from_pretrained(
                        model_config["model_name"]
                    )
                    tokenizer = T5Tokenizer.from_pretrained(model_config["model_name"])
                    model_version = f"pretrained-{model_config['model_name']}"
                    logger.warning(
                        f"Using pretrained model as fallback: {model_config['model_name']}"
                    )

            # Move model to device
            model.to(self.device)
            model.eval()

            # Store model info
            self.models[model_key] = {
                "model": model,
                "tokenizer": tokenizer,
                "model_version": model_version,
                "model_name": model_config["model_name"],
                "run_id": model_config["run_id"],
                "description": model_config.get("description", ""),
                "loaded_at": datetime.now(),
            }

            logger.info(f"Model '{model_key}' loaded successfully on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model '{model_key}': {e}")
            return False

    def load_all_models(self) -> bool:
        """
        Load all configured models.

        Returns:
            bool: True if all models loaded successfully, False otherwise
        """
        success = True
        for model_key in self.config["models"].keys():
            if not self.load_model(model_key):
                success = False
        return success

    def is_model_loaded(self, model_key: str) -> bool:
        """Check if a specific model is loaded."""
        return model_key in self.models and self.models[model_key]["model"] is not None

    def is_any_model_loaded(self) -> bool:
        """Check if any model is loaded."""
        return len(self.models) > 0 and any(
            model_info["model"] is not None for model_info in self.models.values()
        )

    def get_loaded_models(self) -> List[str]:
        """Get list of loaded model keys."""
        return [key for key, info in self.models.items() if info["model"] is not None]

    def get_model_info(self, model_key: str = None) -> Dict[str, Any]:
        """Get model information."""
        if model_key:
            if model_key not in self.models:
                raise ValueError(f"Model '{model_key}' not found")
            model_info = self.models[model_key]
            return {
                "model_key": model_key,
                "model_name": model_info["model_name"],
                "model_version": model_info["model_version"],
                "run_id": model_info["run_id"],
                "description": model_info["description"],
                "device": str(self.device),
                "loaded_at": model_info["loaded_at"],
                "is_loaded": self.is_model_loaded(model_key),
            }
        else:
            # Return info for all models
            return {
                key: {
                    "model_name": info["model_name"],
                    "model_version": info["model_version"],
                    "run_id": info["run_id"],
                    "description": info["description"],
                    "device": str(self.device),
                    "loaded_at": info["loaded_at"],
                    "is_loaded": self.is_model_loaded(key),
                }
                for key, info in self.models.items()
            }

    def get_uptime(self) -> float:
        """Get service uptime in seconds."""
        return time.time() - self.start_time

    def predict_action(
        self,
        mission_state: str,
        model_key: str = None,
        max_length: int = 150,
        num_beams: int = 2,
        temperature: float = 1.0,
        repetition_penalty: float = 2.5,
    ) -> Dict[str, Any]:
        """
        Predict next action for given mission state.

        Args:
            mission_state: Current mission state and context
            model_key: Key of the model to use (if None, uses first available model)
            max_length: Maximum length of generated action
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            repetition_penalty: Repetition penalty

        Returns:
            Dict containing predicted action and metadata
        """
        if not self.is_any_model_loaded():
            raise RuntimeError("No models loaded")

        # Use specified model or first available model
        if model_key is None:
            model_key = self.get_loaded_models()[0]
        elif not self.is_model_loaded(model_key):
            raise RuntimeError(f"Model '{model_key}' not loaded")

        model_info = self.models[model_key]
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]

        start_time = time.time()

        try:
            # Preprocess input - add the prefix used in training
            input_text = f"summarize: {mission_state}"

            # Tokenize input
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            # Generate action
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=max_length,
                    num_beams=num_beams,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    length_penalty=1.0,
                    early_stopping=True,
                    do_sample=temperature != 1.0,
                )

            # Decode output
            predicted_action = tokenizer.decode(
                outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            processing_time = (
                time.time() - start_time
            ) * 1000  # Convert to milliseconds

            return {
                "predicted_action": predicted_action,
                "input_length": len(mission_state),
                "output_length": len(predicted_action),
                "processing_time_ms": processing_time,
                "model_key": model_key,
                "model_version": model_info["model_version"],
                "run_id": model_info["run_id"],
            }

        except Exception as e:
            logger.error(
                f"Error during action prediction with model '{model_key}': {e}"
            )
            raise RuntimeError(f"Action prediction failed: {e}")

    def batch_predict_action(
        self,
        mission_states: list,
        model_key: str = None,
        max_length: int = 150,
        num_beams: int = 2,
        temperature: float = 1.0,
        repetition_penalty: float = 2.5,
    ) -> Dict[str, Any]:
        """
        Predict actions for multiple mission states.

        Args:
            mission_states: List of mission states to predict actions for
            model_key: Key of the model to use (if None, uses first available model)
            max_length: Maximum length of generated actions
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            repetition_penalty: Repetition penalty

        Returns:
            Dict containing predicted actions and metadata
        """
        if not self.is_any_model_loaded():
            raise RuntimeError("No models loaded")

        # Use specified model or first available model
        if model_key is None:
            model_key = self.get_loaded_models()[0]
        elif not self.is_model_loaded(model_key):
            raise RuntimeError(f"Model '{model_key}' not loaded")

        start_time = time.time()
        predicted_actions = []

        try:
            for mission_state in mission_states:
                result = self.predict_action(
                    mission_state,
                    model_key,
                    max_length,
                    num_beams,
                    temperature,
                    repetition_penalty,
                )
                predicted_actions.append(result["predicted_action"])

            processing_time = (time.time() - start_time) * 1000

            model_info = self.models[model_key]
            return {
                "predicted_actions": predicted_actions,
                "processing_time_ms": processing_time,
                "model_key": model_key,
                "model_version": model_info["model_version"],
                "run_id": model_info["run_id"],
            }

        except Exception as e:
            logger.error(
                f"Error during batch action prediction with model '{model_key}': {e}"
            )
            raise RuntimeError(f"Batch action prediction failed: {e}")

    def log_inference_metrics(self, metrics: Dict[str, Any], model_key: str = None):
        """
        Log inference metrics to MLFlow.

        Args:
            metrics: Dictionary containing metrics to log
            model_key: Key of the model used for inference
        """
        try:
            with mlflow.start_run(tags={"inference_test": "true"}):
                # Log model info
                if model_key and model_key in self.models:
                    model_info = self.models[model_key]
                    mlflow.log_param("model_key", model_key)
                    mlflow.log_param("model_run_id", model_info["run_id"])
                    mlflow.log_param("model_version", model_info["model_version"])

                # Log metrics
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
                    else:
                        mlflow.log_param(key, str(value))

                logger.info(
                    f"Inference metrics logged to MLFlow for model '{model_key}'"
                )

        except Exception as e:
            logger.error(f"Failed to log inference metrics: {e}")
