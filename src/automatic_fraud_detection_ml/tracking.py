from dotenv import load_dotenv
import mlflow
from mlflow import MlflowClient
import os
from sqlalchemy.orm import Session


from fraud_detection_database.models.model_version import (
    ModelVersion as DatabaseModelVersion,
)

from mlflow.entities.model_registry import ModelVersion as MlflowModelVersion


def configure_mlflow() -> None:
    load_dotenv("secrets.env")

    mlflow.set_tracking_uri(
        os.environ["MLFLOW_TRACKING_URI"]
    )

    mlflow.set_experiment(
        os.environ["MLFLOW_EXPERIMENT_NAME"]
    )




def register_model_version_in_database(
    session: Session,
    model_name: str,
    model_version: str,
    run_id: str,
    model_uri: str,
    threshold: float,
    feature_version: str,
) -> None:
    model = DatabaseModelVersion(
        model_name=model_name,
        model_version=model_version,
        run_id=run_id,
        model_uri=model_uri,
        threshold=threshold,
        feature_version=feature_version,
    )

    session.add(model)


def get_registered_model_version(
    model_name: str,
    run_id: str,
) -> MlflowModelVersion:
    client = MlflowClient()

    versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    for version in versions:
        if version.run_id == run_id:
            return version

    raise ValueError(
        f"No model version found for model "
        f"{model_name!r} and run {run_id!r}"
    )