import json
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from typing import Tuple, List
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from transformers import T5Tokenizer
from loguru import logger


def split_ids_by_ratio(
    unique_ids: List,
    train_ratio: float = 0.5,
    val_ratio: float = 0.25,
    test_ratio: float = 0.25,
    random_seed: int = 42,
) -> Tuple[List, List, List]:
    """
    Splits a list of unique IDs into train, validation, and test sets based on ratios.

    Args:
        unique_ids: A list of unique identifiers.
        train_ratio: The proportion of IDs for the training set.
        val_ratio: The proportion of IDs for the validation set.
        test_ratio: The proportion of IDs for the test set.
        random_seed: Random seed for reproducible shuffling.

    Returns:
        A tuple containing three lists: (train_ids, val_ids, test_ids).
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("The sum of train, val, and test ratios must be 1.0")

    np.random.seed(random_seed)
    shuffled_ids = np.random.permutation(unique_ids)

    total_ids = len(shuffled_ids)
    train_end = int(total_ids * train_ratio)
    val_end = train_end + int(total_ids * val_ratio)

    train_ids = shuffled_ids[:train_end].tolist()
    val_ids = shuffled_ids[train_end:val_end].tolist()
    test_ids = shuffled_ids[val_end:].tolist()

    return train_ids, val_ids, test_ids


class DroneLogsDataset(Dataset):
    def __init__(self, data_folder: Path, annotations_folder: Path):
        self.data_folder = data_folder
        self.annotations = self._read_annotations(annotations_folder)

    def _read_annotations(self, labels_folder: Path):
        """Read all JSON files from the given folder path"""
        if not labels_folder.exists():
            raise FileNotFoundError(f"Folder not found: {labels_folder}")
        label_data = []
        for json_path in labels_folder.iterdir():
            label_data.append(self._read_json(json_path))
        return pd.DataFrame(label_data)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations.iloc[idx]
        path_to_sample = self.data_folder / Path(annotation.path).name
        sample = self._read_json(path_to_sample)
        return sample.get("input"), sample.get("output")

    def _read_json(self, json_path: Path):
        if json_path.is_file() and json_path.suffix == ".json":
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        else:
            raise ValueError(f"Invalid JSON file: {json_path}")


def create_dataloaders(cfg: DictConfig, tokenizer: T5Tokenizer):
    """
    Creates and returns the training and validation dataloaders.
    """
    ANNOTATIONS_PATH = Path(cfg.data.annotations_path)
    DATA_FOLDER_PATH = Path(cfg.data.data_folder_path)
    RANDOM_SEED = cfg.training.random_seed
    BATCH_SIZE = cfg.data.batch_size

    full_dataset = DroneLogsDataset(DATA_FOLDER_PATH, ANNOTATIONS_PATH)

    if "id_in_db" not in full_dataset.annotations.columns:
        raise ValueError(
            "'id_in_db' column not found in annotations. Cannot split data."
        )

    unique_ids = full_dataset.annotations["id_in_db"].unique()

    train_ids, val_ids, test_ids = split_ids_by_ratio(
        unique_ids,
        train_ratio=cfg.data.split_ratio.train,
        val_ratio=cfg.data.split_ratio.val,
        test_ratio=cfg.data.split_ratio.test,
        random_seed=RANDOM_SEED,
    )

    train_annotations = full_dataset.annotations[
        full_dataset.annotations["id_in_db"].isin(train_ids)
    ]
    val_annotations = full_dataset.annotations[
        full_dataset.annotations["id_in_db"].isin(val_ids)
    ]
    test_annotations = full_dataset.annotations[
        full_dataset.annotations["id_in_db"].isin(test_ids)
    ]

    def collate_fn(batch):
        inputs, outputs = zip(*batch)

        prefixed_inputs = [f"summarize: {text}" for text in inputs]

        input_encodings = tokenizer(
            prefixed_inputs, padding=True, truncation=True, return_tensors="pt"
        )
        output_encodings = tokenizer(
            outputs, padding=True, truncation=True, return_tensors="pt"
        )

        labels = output_encodings.input_ids
        labels[labels == tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_encodings.input_ids,
            "attention_mask": input_encodings.attention_mask,
            "labels": labels,
            "original_input": inputs,
            "original_output": outputs,
        }

    train_dataset = DroneLogsDataset(DATA_FOLDER_PATH, ANNOTATIONS_PATH)
    train_dataset.annotations = train_annotations.reset_index(drop=True)

    val_dataset = DroneLogsDataset(DATA_FOLDER_PATH, ANNOTATIONS_PATH)
    val_dataset.annotations = val_annotations.reset_index(drop=True)

    test_dataset = DroneLogsDataset(DATA_FOLDER_PATH, ANNOTATIONS_PATH)
    test_dataset.annotations = test_annotations.reset_index(drop=True)

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn
    )
    test_dataset = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn
    )
    return train_dataloader, val_dataloader, test_dataset


if __name__ == "__main__":
    path_to_annotations = Path(".processed_samples/dataset/annotations")
    path_to_data_folder = Path(".processed_samples/dataset/samples")
    dataset = DroneLogsDataset(path_to_data_folder, path_to_annotations)
    logger.info(f"Sample dataset item: {dataset[1]}")
