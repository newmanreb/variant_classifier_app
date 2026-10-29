import io
import pytest
from app import app


@pytest.fixture
def client():
    """Create a test client for the Flask application."""
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


def test_homepage_loads(client):
    """The homepage should load successfully."""
    response = client.get("/")

    assert response.status_code == 200


def test_single_prediction_returns_result(client):
    """A valid variant submission should return a prediction."""
    response = client.post(
        "/predict",
        data={
            "mc_variant": "MC_frameshift_variant",
            "gene_length": "10000",
            "pli_score": "0.95",
            "rs_present": "1",
            "allele_frequency": "0.001",
            "variant_type": "Deletion"
        }
    )

    assert response.status_code == 200
    assert b"Pathogenic" in response.data or b"Benign" in response.data


def test_missing_batch_file_returns_error(client):
    """Submitting a batch request without a file should show an error."""
    response = client.post("/predict_batch", data={})

    assert response.status_code == 200
    assert b"No file selected" in response.data


def test_batch_prediction_accepts_csv(client):
    """A valid CSV upload should produce batch results."""
    csv_content = (
        "mc_variant,Gene_Length,pLI_Score,RS_Present,"
        "Allele_Frequency,Type\n"
        "SO:0001583,10000,0.95,1,0.001,missense\n"
    )

    response = client.post(
        "/predict_batch",
        data={
            "file": (
                io.BytesIO(csv_content.encode("utf-8")),
                "test_variants.csv"
            )
        },
        content_type="multipart/form-data"
    )

    assert response.status_code == 200
    assert b"SO:0001583" in response.data
