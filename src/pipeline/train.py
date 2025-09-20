import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.optim import AdamW
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_scheduler
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import mlflow
import mlflow.pytorch
import json
from loguru import logger
from src.pipeline.dataset import create_dataloaders


@hydra.main(config_path="configs", config_name="train", version_base=None)
def train_model(cfg: DictConfig):
    """
    Main function to train the T5 model on the DroneLogs dataset.
    """
    logger.info(OmegaConf.to_yaml(cfg))
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    MODEL_NAME = cfg.model.name
    LEARNING_RATE = cfg.training.learning_rate
    NUM_EPOCHS = cfg.training.num_epochs
    RANDOM_SEED = cfg.training.random_seed
    MODEL_SAVE_PATH = Path(cfg.model.save_path)
    MODEL_SAVE_PATH.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(device)

    train_dataloader, val_dataloader, _ = create_dataloaders(cfg, tokenizer)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    with mlflow.start_run(
        run_name=f"{MODEL_NAME}-run-{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}"
    ):
        mlflow.log_params(
            {
                "model_name": MODEL_NAME,
                "learning_rate": LEARNING_RATE,
                "batch_size": cfg.data.batch_size,
                "num_epochs": NUM_EPOCHS,
                "random_seed": RANDOM_SEED,
            }
        )

        for epoch in range(NUM_EPOCHS):
            logger.info(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")

            model.train()
            total_train_loss = 0
            for batch in tqdm(train_dataloader, desc="Training"):
                input_batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                outputs = model(**input_batch)
                loss = outputs.loss

                total_train_loss += loss.item()

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            avg_train_loss = total_train_loss / len(train_dataloader)
            logger.info(f"Average Training Loss: {avg_train_loss:.4f}")
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validation"):
                    input_batch = {
                        k: v.to(device)
                        for k, v in batch.items()
                        if isinstance(v, torch.Tensor)
                    }
                    outputs = model(**input_batch)
                    loss = outputs.loss
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_dataloader)
            logger.info(f"Average Validation Loss: {avg_val_loss:.4f}")
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

        logger.info("Training complete. Logging model with MLFlow...")
        # mlflow.pytorch.log_model(model, "model")
        mlflow.log_artifacts(str(MODEL_SAVE_PATH), artifact_path="model_and_tokenizer")

        # Save model and tokenizer locally
        model.save_pretrained(MODEL_SAVE_PATH)
        tokenizer.save_pretrained(MODEL_SAVE_PATH)
        mlflow.log_artifacts(str(MODEL_SAVE_PATH), artifact_path="tokenizer")

        # Save training metrics to JSON file
        metrics_path = Path("metrics")
        metrics_path.mkdir(exist_ok=True)

        training_metrics = {
            "final_train_loss": avg_train_loss,
            "final_val_loss": avg_val_loss,
            "model_name": MODEL_NAME,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "batch_size": cfg.data.batch_size,
            "random_seed": RANDOM_SEED,
        }

        with open("metrics/training_metrics.json", "w") as f:
            json.dump(training_metrics, f, indent=2)

        logger.info(f"Model and tokenizer logged to MLFlow.")
        logger.info(f"Training metrics saved to metrics/training_metrics.json")


if __name__ == "__main__":
    train_model()
