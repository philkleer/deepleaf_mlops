import os
import pytest
from fastapi.testclient import TestClient
from app.main import app  # Import the FastAPI app from your app.main module

# Create a test client instance
client = TestClient(app)

# Test 1: Check if the dataset file exists
def test_dataset_exists():
    dataset_path = "data/raw/train_subset1.tfrecord"  # Path to the dataset
    assert os.path.exists(dataset_path), f"Dataset not found: {dataset_path}"

# Test 2: Test the /train API endpoint - valid dataset path
def test_train_model_success():
    dataset_path = "data/raw/train_subset1.tfrecord"  # Path to the dataset
    response = client.post("/train", json={"dataset_path": dataset_path})
    
    # Assert that the response status code is 200
    assert response.status_code == 200
    # Assert that the response body contains the expected message
    assert response.json() == {"message": "Model training started."}

# Test 3: Test the /train API endpoint - invalid dataset path
def test_train_model_file_not_found():
    invalid_dataset_path = "data/raw/non_existing_dataset.tfrecord"  # Invalid dataset path
    response = client.post("/train", json={"dataset_path": invalid_dataset_path})
    
    # Assert that the response status code is 500 (Internal Server Error) when the file doesn't exist
    assert response.status_code == 500
    # Optionally, you can check the error message, depending on how you've implemented error handling
    assert "detail" in response.json()  # Checking if error detail exists in the response
