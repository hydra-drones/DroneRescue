import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import mlflow
from dataset import create_dataloaders


@hydra.main(config_path="configs", config_name="test", version_base=None)
def test_model(cfg: DictConfig):
    """
    Main function to test the T5 model on the DroneLogs dataset.
    """
    print(OmegaConf.to_yaml(cfg))
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    MODEL_NAME = cfg.model.name
    MODEL_PATH = cfg.model.path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    model.to(device)

    _, _, test_dataloader = create_dataloaders(cfg, tokenizer)

    model.eval()
    results = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=150,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
            )

            preds = [
                tokenizer.decode(
                    g, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for g in generated_ids
            ]

            for i in range(len(preds)):
                results.append(
                    {
                        "input": batch["original_input"][i],
                        "generated_summary": preds[i],
                        "original_summary": batch["original_output"][i],
                    }
                )

    results_df = pd.DataFrame(results)
    results_df.to_csv("test_results.csv", index=False)
    print("Testing complete. Results saved to test_results.csv")


if __name__ == "__main__":
    test_model()
