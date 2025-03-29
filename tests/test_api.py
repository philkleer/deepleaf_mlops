import pytest
from fastapi.testclient import TestClient
from PIL import Image
from io import BytesIO
from app.main import app

client = TestClient(app)

def test_no_file_uploaded():
    """
    Scenario 1: No file is uploaded at all.
    The endpoint should return a 400 with "No file uploaded."
    """
    response = client.post("/predict/")
    assert response.status_code == 400
    assert "No file uploaded" in response.text

def test_invalid_extension():
    """
    Scenario 2: Invalid file extension. 
    The endpoint checks extension and should return 400 if not jpg/jpeg/png.
    """
    file_data = b"Invalid content for testing extension"
    files = {"file": ("test.txt", file_data, "text/plain")}
    response = client.post("/predict/", files=files)
    assert response.status_code == 400
    assert "Invalid file type" in response.text

def test_file_too_large():
    """
    Scenario 3: File size exceeding the 5MB threshold.
    Should return HTTP 413 (Payload Too Large).
    """
    # Generate ~6MB of data
    large_content = b"X" * (6 * 1024 * 1024)
    files = {"file": ("big_test.jpg", large_content, "image/jpeg")}
    response = client.post("/predict/", files=files)
    assert response.status_code == 413
    assert "exceeds" in response.text

@pytest.fixture
def valid_jpg_fixture(tmp_path):
    """
    Creates a small valid .jpg file using Pillow for a 'happy path' test.
    """
    file_path = tmp_path / "test_valid.jpg"
    img = Image.new("RGB", (1, 1))  # 1×1 pixel image
    img.save(file_path, format="JPEG")
    return file_path

def test_valid_jpg(valid_jpg_fixture):
    """
    Scenario 4: Valid .jpg upload that should pass all checks.
    Expects a 200 status code and a JSON response containing "prediction".
    """
    with open(valid_jpg_fixture, "rb") as f:
        files = {"file": ("test_valid.jpg", f, "image/jpeg")}
        response = client.post("/predict/", files=files)
    assert response.status_code == 200, f"Expected 200, got {response.status_code} - {response.text}"
    json_data = response.json()
    assert "prediction" in json_data, f"Response JSON: {json_data}"