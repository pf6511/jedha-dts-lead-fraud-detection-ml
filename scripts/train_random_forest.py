import mlflow
import mlflow.sklearn
import os
import time

from fraud_detection_features.config import (
    TIMESTAMP_COLUMN,
)

from automatic_fraud_detection_ml.config import (
    DATASET_TRAIN_RATIO,
    DATASET_VAL_RATIO,
    DATASET_TEST_RATIO
)

from fraud_detection_features.features import build_training_features

from automatic_fraud_detection_ml.model import  (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_TARGET_COLUMN,
    RANDOM_FOREST_PARAMS,
    build_model
)
from automatic_fraud_detection_ml.training import train_model
from automatic_fraud_detection_ml.tracking import (
    configure_mlflow,
    get_registered_model_version,
    register_model_version_in_database)

from automatic_fraud_detection_ml.evaluation import evaluate_thresholds, evaluate_at_threshold

from automatic_fraud_detection_ml.business import FraudDetectionBusinessRules

from fraud_detection_database.database import Database
from fraud_detection_database.environment import Environment

from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/fraudTest.csv")

THRESHOLDS = [
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
]


RUN_NAME = "random-forest-card-history"

def get_database_access() -> Database:
    environment = Environment("./secrets.env")
    return Database(
        database_url=environment.env_var("DATABASE_URL"),
        admin_database_url=environment.env_var("ADMIN_DATABASE_URL"),
    )

def main() -> None:
    configure_mlflow()


    # MLflow

    with mlflow.start_run(run_name=RUN_NAME) as run:
        print("start experience...")
        start_time = time.time()
        # Load
        print("load data...")
        dtf = pd.read_csv(DATA_PATH)
        dtf[TIMESTAMP_COLUMN] = pd.to_datetime(dtf[TIMESTAMP_COLUMN])
        # sort according to transaction date time
        dtf = dtf.sort_values(
            TIMESTAMP_COLUMN
        ).reset_index(drop=True)

        # Split
        n = len(dtf)

        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        dtf_train = dtf.iloc[:train_end]
        p0 = dtf_train[MODEL_TARGET_COLUMN].mean()

        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "n_estimators": RANDOM_FOREST_PARAMS["n_estimators"],
                "class_weight": RANDOM_FOREST_PARAMS["class_weight"],
                "random_state": RANDOM_FOREST_PARAMS["random_state"],
                "alpha": FraudDetectionBusinessRules.ALPHA,
                "p0": p0,
                "threshold": FraudDetectionBusinessRules.FRAUD_THRESHOLD,
                "split_strategy": "chronological",
                "train_ratio": DATASET_TRAIN_RATIO,
                "validation_ratio": DATASET_VAL_RATIO,
                "test_ratio": DATASET_TEST_RATIO,
                "feature_set": "card_history_v2",
            }
        )

        # Feature engineering
        print("feature engineering..")
        dtf = build_training_features(
            dtf,
            p0=p0,
            alpha=FraudDetectionBusinessRules.ALPHA,
        )
        #print(dtf.isna().sum().loc[lambda x: x > 0])
        
        dtf_train = dtf.iloc[:train_end]
        dtf_val = dtf.iloc[train_end:val_end]
        dtf_test = dtf.iloc[val_end:]

        # Features
        print("split dataset..")
        features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

        X_train = dtf_train[features]
        y_train = dtf_train[MODEL_TARGET_COLUMN]

        X_val = dtf_val[features]
        y_val = dtf_val[MODEL_TARGET_COLUMN]

        X_test = dtf_test[features]
        y_test = dtf_test[MODEL_TARGET_COLUMN]

        # Model
        print("build model..")
        model = build_model()

        print("train model..")
        model = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
        )

        # Validation
        print("evaluate model..")
        val_probabilities = model.predict_proba(
            X_val
        )[:, 1]

        threshold_results = evaluate_thresholds(
            y_true=y_val,
            probabilities=val_probabilities,
            thresholds=THRESHOLDS,
        )

        val_metrics = evaluate_at_threshold(
            y_true=y_val,
            probabilities=val_probabilities,
            threshold=FraudDetectionBusinessRules.FRAUD_THRESHOLD,
        )


        # Test
        test_probabilities = model.predict_proba(
            X_test
        )[:, 1]

        test_metrics = evaluate_at_threshold(
            y_true=y_test,
            probabilities=test_probabilities,
            threshold=FraudDetectionBusinessRules.FRAUD_THRESHOLD,
        )

        print("log to mlflow..")
        mlflow.log_metrics(
            {
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_TP": test_metrics["TP"],
                "test_FP": test_metrics["FP"],
                "test_FN": test_metrics["FN"],
                "test_TN": test_metrics["TN"],
            }
        )

        threshold_results.to_csv(
            "threshold_metrics.csv",
            index=False,
        )

        mlflow.log_artifact(
            "threshold_metrics.csv",
        )

        mlflow.sklearn.log_model(
            model,
            name="fraud_random_forest",
            registered_model_name="fraud-detection-random-forest",
            serialization_format="skops",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
            ],           
        )

        registered_model_version = get_registered_model_version(
            model_name="fraud-detection-random-forest",
            run_id=run.info.run_id,
        )

        with get_database_access().AdminSessionLocal() as session:
            try:
                register_model_version_in_database(
                    session=session,
                    model_name="fraud-detection-random-forest",
                    model_version=registered_model_version.version,
                    run_id=run.info.run_id,
                    model_uri=registered_model_version.source,
                    threshold=FraudDetectionBusinessRules.FRAUD_THRESHOLD,
                    feature_version="card_history_v2",
                )

                session.commit()

            except Exception:
                session.rollback()
                raise

        print("Validation metrics:")
        print(val_metrics)

        print("\nTest metrics:")
        print(test_metrics)

        print("Logged model and metrics to MLflow.")
        print("Model trained and logged to:", run.info.artifact_uri)


if __name__ == "__main__":
    main()