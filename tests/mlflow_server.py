import pytest
import requests

# URL of your MLflow server
MLFLOW_URI = "http://localhost:5001"

def test_mlflow_server():
    # Send a GET request to the MLflow server's tracking API (list experiments as a test)
    try:
        response = requests.get(f"{MLFLOW_URI}/api/2.0/mlflow/experiments/list")
        
        # Assert that the response status code is 200, which means the server is up
        assert response.status_code == 200
        
        # Optionally, check if the response contains a valid list (to ensure the server is functional)
        data = response.json()
        assert isinstance(data, dict)  # Ensure the response is a dictionary (could also validate keys if needed)

    except requests.exceptions.RequestException as e:
        pytest.fail(f"MLflow server is not running. Error: {e}")