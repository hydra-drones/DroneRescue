# Snakemake pipeline for DroneRescue ML project
# This pipeline orchestrates data preprocessing, model training, and evaluation
# using the existing train.py, test.py, and dataset_cli.py functions
# with DVC integration for data versioning

# Configuration
configfile: "config.yaml"

# Define all target files
rule all:
    input:
        "metrics/evaluation_results.csv",
        "models/trained_model"

# Start MLflow server
rule start_mlflow:
    output:
        mlflow_ready=".mlflow_server_ready"
    shell:
        """
        poetry install
        poetry run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 &
        echo "MLflow server started" > .mlflow_server_ready
        sleep 5
        """

# Data preprocessing step using the existing dataset_cli function
rule preprocess:
    input:
        db_file=".database/data.db"
    output:
        processed_data=directory(".processed_samples/dataset")
    shell:
        """
        poetry install
        poetry run drone-rescue data process-dataset src.dataset.dataset_v1.AlpacaDatasetV1 \
            ./.processed_samples/dataset -d .database/data.db
        """

# Model training step using the existing train.py
rule train:
    input:
        annotations=".processed_samples/dataset/annotations",
        samples=".processed_samples/dataset/samples",
        mlflow_ready=".mlflow_server_ready"
    output:
        model_dir=directory("models/trained_model"),
        training_metrics="metrics/training_metrics.json"
    shell:
        """
        poetry install
        poetry run python src/pipeline/train.py
        """

# Model evaluation step using the existing test.py
rule evaluate:
    input:
        model_dir="models/trained_model",
        annotations=".processed_samples/dataset/annotations",
        samples=".processed_samples/dataset/samples",
        mlflow_ready=".mlflow_server_ready"
    output:
        results="metrics/evaluation_results.csv"
    shell:
        """
        poetry install
        poetry run python src/pipeline/test.py
        """

# DVC integration rules
rule dvc_pull:
    shell:
        """
        poetry run dvc pull
        """

rule dvc_add:
    shell:
        """
        poetry run dvc add .processed_samples/dataset
        poetry run dvc add models/trained_model
        poetry run dvc add metrics/
        """

rule dvc_push:
    shell:
        """
        poetry run dvc push
        """

# Clean up rule
rule clean:
    shell:
        """
        rm -rf .processed_samples/dataset/*
        rm -rf models/trained_model/*
        rm -rf metrics/*
        rm -f .mlflow_server_ready
        pkill -f "mlflow server" || true
        """

# Stop MLflow server
rule stop_mlflow:
    shell:
        """
        pkill -f "mlflow server" || true
        rm -f .mlflow_server_ready
        """
