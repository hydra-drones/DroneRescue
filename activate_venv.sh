#!/bin/bash

# Define variables
ENVIRONMENT_NAME="venv"
PYTHON_VERSION="python3.10"

# Check if the required Python version is installed
if ! command -v $PYTHON_VERSION &> /dev/null; then
    echo "Error: $PYTHON_VERSION is not installed. Please install it before running this script."
    exit 1
fi


# Create the virtual environment using Python
$PYTHON_VERSION -m venv $ENVIRONMENT_NAME

# Check if the virtual environment was created successfully
if [ ! -d "$ENVIRONMENT_NAME" ]; then
    echo "Error: Failed to create virtual environment."
    exit 1
fi

# Activate the virtual environment
source "$ENVIRONMENT_NAME/bin/activate"

pip install poetry

# Use Poetry to install dependencies
if [ -f "pyproject.toml" ]; then
    poetry install
    echo "Successfully installed dependencies using Poetry."
else
    echo "No pyproject.toml file found. Skipping dependency installation."
fi
