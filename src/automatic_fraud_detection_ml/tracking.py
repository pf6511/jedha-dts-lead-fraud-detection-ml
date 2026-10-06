from dotenv import load_dotenv
import mlflow
import os


def configure_mlflow() -> None:
    load_dotenv("secrets.env")

    mlflow.set_tracking_uri(
        os.environ["MLFLOW_TRACKING_URI"]
    )

    mlflow.set_experiment(
        os.environ["MLFLOW_EXPERIMENT_NAME"]
    )