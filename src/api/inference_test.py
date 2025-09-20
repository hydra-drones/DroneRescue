"""
Simple function to test inference API and save results to MLFlow.
"""
import requests
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from loguru import logger
import mlflow
import yaml
from transformers import T5Tokenizer
from src.pipeline.dataset import create_dataloaders
from omegaconf import OmegaConf
import numpy as np
from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate evaluation metrics for the inference results.

    Args:
        results: List of inference results

    Returns:
        Dictionary containing calculated metrics
    """
    if not results:
        return {}

    # Extract predictions and ground truth
    predictions = [r["predicted_output"] for r in results if r["predicted_output"]]
    ground_truth = [r["expected_output"] for r in results if r["predicted_output"]]

    if not predictions:
        return {"error": "No successful predictions"}

    # Basic metrics
    total_samples = len(results)
    successful_predictions = len(predictions)
    success_rate = successful_predictions / total_samples

    # Processing time metrics
    processing_times = [
        r["processing_time_ms"] for r in results if r["processing_time_ms"]
    ]
    avg_processing_time = np.mean(processing_times) if processing_times else 0
    min_processing_time = np.min(processing_times) if processing_times else 0
    max_processing_time = np.max(processing_times) if processing_times else 0

    # Text similarity metrics
    bleu_scores = []
    exact_matches = 0

    smoothing = SmoothingFunction().method1

    for pred, truth in zip(predictions, ground_truth):
        # Exact match
        if pred.strip().lower() == truth.strip().lower():
            exact_matches += 1

        # BLEU score
        try:
            pred_tokens = nltk.word_tokenize(pred.lower())
            truth_tokens = nltk.word_tokenize(truth.lower())
            bleu_score = sentence_bleu(
                [truth_tokens], pred_tokens, smoothing_function=smoothing
            )
            bleu_scores.append(bleu_score)
        except:
            bleu_scores.append(0.0)

    # Calculate averages
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
    exact_match_rate = exact_matches / len(predictions) if predictions else 0

    # Length statistics
    pred_lengths = [len(pred.split()) for pred in predictions]
    truth_lengths = [len(truth.split()) for truth in ground_truth]

    avg_pred_length = np.mean(pred_lengths) if pred_lengths else 0
    avg_truth_length = np.mean(truth_lengths) if truth_lengths else 0

    return {
        "total_samples": total_samples,
        "successful_predictions": successful_predictions,
        "success_rate": success_rate,
        "exact_match_rate": exact_match_rate,
        "avg_bleu_score": avg_bleu,
        "avg_processing_time_ms": avg_processing_time,
        "min_processing_time_ms": min_processing_time,
        "max_processing_time_ms": max_processing_time,
        "avg_prediction_length": avg_pred_length,
        "avg_ground_truth_length": avg_truth_length,
        "bleu_scores": bleu_scores,
        "processing_times": processing_times,
    }


def run_inference_test(
    config_path: str = "src/api/config.yaml", model_key: str = None
) -> Dict[str, Any]:
    """
    Run inference test using API config and save results to MLFlow.

    Args:
        config_path: Path to the API configuration file
        model_key: Key of the model to use for inference (if None, uses first model)

    Returns:
        Dictionary containing test results
    """
    logger.info(f"Starting inference test with config: {config_path}")

    # Load API config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize MLFlow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Get model info from config
    if model_key is None:
        model_key = list(config["models"].keys())[0]  # Use first model if not specified
    elif model_key not in config["models"]:
        available_models = list(config["models"].keys())
        raise ValueError(
            f"Model '{model_key}' not found in config. Available models: {available_models}"
        )

    run_id = config["models"][model_key]["run_id"]

    logger.info(f"Using model '{model_key}' with run ID: {run_id}")

    # Load test samples from dataloader (like in test.py)
    logger.info("Loading test samples from dataloader")

    # Load tokenizer from the same run ID as the model
    artifacts_base = Path("mlartifacts")
    tokenizer = None

    # Search for tokenizer in mlartifacts
    for exp_dir in artifacts_base.iterdir():
        if exp_dir.is_dir():
            run_dir = exp_dir / run_id
            if run_dir.exists():
                model_tokenizer_dir = run_dir / "artifacts" / "model_and_tokenizer"
                if model_tokenizer_dir.exists():
                    logger.info(f"Found tokenizer directory at: {model_tokenizer_dir}")
                    tokenizer = T5Tokenizer.from_pretrained(str(model_tokenizer_dir))
                    break

    if tokenizer is None:
        logger.warning(f"Tokenizer not found for run {run_id}, using pretrained")
        tokenizer = T5Tokenizer.from_pretrained(
            config["models"][model_key]["model_name"]
        )

    cfg = OmegaConf.load("src/pipeline/configs/test.yaml")

    _, _, test_dataloader = create_dataloaders(cfg, tokenizer)

    # Collect test samples from dataloader
    test_samples = []
    max_samples = 10  # Limit samples for API testing

    sample_count = 0
    for batch in test_dataloader:
        if sample_count >= max_samples:
            break

        for i in range(len(batch["original_input"])):
            if sample_count >= max_samples:
                break

            test_samples.append(
                {
                    "input": batch["original_input"][i],
                    "expected_output": batch["original_output"][i],
                }
            )
            sample_count += 1

    logger.info(f"Loaded {len(test_samples)} test samples from dataloader")

    api_url = f"http://localhost:{config['server']['port']}"
    results = []

    logger.info(f"Sending {len(test_samples)} test samples to API")

    for i, sample in enumerate(test_samples):
        try:
            payload = {
                "mission_state": sample["input"],
                "model_key": model_key,
                "max_length": config["inference"]["max_length"],
                "num_beams": config["inference"]["num_beams"],
                "temperature": config["inference"]["temperature"],
                "repetition_penalty": config["inference"]["repetition_penalty"],
            }

            response = requests.post(f"{api_url}/predict", json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                results.append(
                    {
                        "sample_index": i,
                        "input": sample["input"],
                        "expected_output": sample["expected_output"],
                        "predicted_output": result["predicted_action"],
                        "processing_time_ms": result["processing_time_ms"],
                        "model_key": result["model_key"],
                        "run_id": result["run_id"],
                    }
                )
                logger.info(f"Sample {i+1}/{len(test_samples)} processed successfully")
            else:
                logger.error(
                    f"API request failed for sample {i+1}: {response.status_code}"
                )

        except Exception as e:
            logger.error(f"Error processing sample {i+1}: {e}")

    # Calculate comprehensive metrics
    metrics = calculate_metrics(results)

    logger.info(
        f"Inference test completed: {metrics['successful_predictions']}/{metrics['total_samples']} successful"
    )
    logger.info(f"Success rate: {metrics['success_rate']:.2%}")
    logger.info(f"Exact match rate: {metrics['exact_match_rate']:.2%}")
    logger.info(f"Average BLEU score: {metrics['avg_bleu_score']:.4f}")
    logger.info(f"Average processing time: {metrics['avg_processing_time_ms']:.2f}ms")

    # Save to MLFlow artifacts
    with mlflow.start_run(tags={"inference_test": "true"}):
        # Log parameters
        mlflow.log_param("model_key", model_key)
        mlflow.log_param("run_id", run_id)
        mlflow.log_param("total_samples", metrics["total_samples"])

        # Log all metrics
        mlflow.log_metric("successful_predictions", metrics["successful_predictions"])
        mlflow.log_metric("success_rate", metrics["success_rate"])
        mlflow.log_metric("exact_match_rate", metrics["exact_match_rate"])
        mlflow.log_metric("avg_bleu_score", metrics["avg_bleu_score"])
        mlflow.log_metric("avg_processing_time_ms", metrics["avg_processing_time_ms"])
        mlflow.log_metric("min_processing_time_ms", metrics["min_processing_time_ms"])
        mlflow.log_metric("max_processing_time_ms", metrics["max_processing_time_ms"])
        mlflow.log_metric("avg_prediction_length", metrics["avg_prediction_length"])
        mlflow.log_metric("avg_ground_truth_length", metrics["avg_ground_truth_length"])

        # Save results as artifact
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as temp_dir:
            results_df = pd.DataFrame(results)
            temp_csv_path = os.path.join(temp_dir, "inference_results.csv")
            results_df.to_csv(temp_csv_path, index=False)
            mlflow.log_artifact(temp_csv_path, "inference_results")

        logger.info("Results saved to MLFlow artifacts")

    return {
        "metrics": metrics,
        "results": results,
        "model_key": model_key,
        "run_id": run_id,
    }


if __name__ == "__main__":
    import sys

    # Allow model selection via command line argument
    if len(sys.argv) > 1:
        model_key = sys.argv[1]
        logger.info(f"Running inference test with model: {model_key}")
        run_inference_test(model_key=model_key)
    else:
        logger.info("Running inference test with default model")
        run_inference_test()
