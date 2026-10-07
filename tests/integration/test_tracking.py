import mlflow

from mlflow import MlflowClient

from automatic_fraud_detection_ml.tracking import ( configure_mlflow
)

from automatic_fraud_detection_ml.tracking import (
    get_registered_model_version,
    register_model_version_in_database
)

from fraud_detection_database.database import AdminSessionLocal

from fraud_detection_database.models.model_version import (
    ModelVersion as DatabaseModelVersion,
)

from sqlalchemy import select


MODEL_NAME = "fraud-detection-random-forest"


def test_get_registered_model_version():
    configure_mlflow()
    client = MlflowClient()

    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    assert versions

    expected = versions[0]

    model_version = get_registered_model_version(
        model_name=MODEL_NAME,
        run_id=expected.run_id,
    )

    assert model_version.version == expected.version
    assert model_version.run_id == expected.run_id
    assert model_version.name == MODEL_NAME


def test_register_model_version_in_database():
    with AdminSessionLocal() as session:
        try:
            register_model_version_in_database(
                session=session,
                model_name="fraud-detection-random-forest",
                model_version="999",
                run_id="test-run-id",
                model_uri="runs:/test-run-id/fraud_random_forest",
                threshold=0.5,
                feature_version="card_history_v2",
            )

            session.flush()

            statement = (
                select(DatabaseModelVersion)
                .where(
                    DatabaseModelVersion.model_name
                    == "fraud-detection-random-forest",
                    DatabaseModelVersion.model_version == "999",
                )
            )

            model = session.scalar(statement)

            assert model is not None
            assert model.run_id == "test-run-id"
            assert model.model_uri == "runs:/test-run-id/fraud_random_forest"
            assert model.threshold == 0.5
            assert model.feature_version == "card_history_v2"

        finally:
            session.rollback()