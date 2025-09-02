import typer
import importlib

from typing import cast
from pathlib import Path
from loguru import logger
from src.dataset.base.processors import BaseDataProcessor
from src.database.scripts import connect_to_db


def process_dataset_cli(
    dataset_object: str = typer.Argument(
        ...,
        help="Python path to the dataset class (e.g., 'src.dataset.dataset_v1.AlpacaDatasetV1')",
    ),
    save_dataset_to_folder: str = typer.Argument(
        ..., help="Folder path to save the processed dataset"
    ),
    db_path: str = typer.Option(
        ".database/data.db", "--db-path", "-d", help="Path to the database file"
    ),
    annotation_path: str = typer.Option(
        None,
        "--annotation-path",
        "-a",
        help="Path for annotation metadata (deprecated)",
    ),
):
    """
    Process dataset using a specified dataset processor class.

    This command dynamically loads a dataset processor class and processes
    drone rescue simulation data from the database into the specified format.

    Example:
        python -m src.cli data process-dataset src.dataset.dataset_v1.AlpacaDatasetV1 ./dataset -d .database/data.db
    """
    if annotation_path is not None:
        logger.warning("annotation_path parameter is deprecated and will be ignored")
        annotation_path = None

    try:
        module_path, class_name = dataset_object.rsplit(".", 1)
        module = importlib.import_module(module_path)
        dataset_class = getattr(module, class_name)

        if not issubclass(dataset_class, BaseDataProcessor):
            raise ImportError(
                "Imported Dataset Processor should be inherited from class :class:`BaseDataProcessor`"
            )
        else:
            data_processor: type[BaseDataProcessor] = cast(
                type[BaseDataProcessor], dataset_class
            )

        db_path_obj = Path(db_path)
        save_folder_obj = Path(save_dataset_to_folder)

        if not db_path_obj.exists():
            raise FileNotFoundError(f"Database file not found at {db_path_obj}")

        save_folder_obj.mkdir(parents=True, exist_ok=True)
        session = connect_to_db(db_path_obj)
        dataset = data_processor(save_folder_obj, annotation_path, session)
        dataset.process_all_samples_in_db()

    except ImportError as e:
        logger.error(f"Failed to import dataset module '{module_path}': {e}")
        raise
    except AttributeError as e:
        logger.error(
            f"Dataset class '{class_name}' not found in module '{module_path}': {e}"
        )
        raise
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        raise
