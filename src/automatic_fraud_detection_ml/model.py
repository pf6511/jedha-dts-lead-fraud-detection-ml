from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from fraud_detection_features.config import (
    TARGET_COLUMN,
    AMOUNT_COLUMN,
    CLIENT_LOCATION_LAT_COLUMN,
    CLIENT_LOCATION_LON_COLUMN,
    CLIENT_LOCATION_CITY_POP_COLUMN,
    MERCHANT_LOCATION_LAT_COLUMN,
    MERCHANT_LOCATION_LON_COLUMN,
    DISTANCE_FEATURE,
    CARD_TRANSACTION_COUNT_FEATURE,
    CARD_FRAUD_COUNT_FEATURE,
    CARD_FRAUD_RATE_FEATURE,
    CARD_AVG_AMOUNT_FEATURE,
    MERCHANT_COLUMN,
    PAYMENT_CATEGORY_COLUMN,
    CLIENT_GENDER_COLUMN,
    CLIENT_LOCATION_STATE_COLUMN,
    CLIENT_JOB_COLUMN
)


NUMERIC_FEATURES = [
    AMOUNT_COLUMN,
    CLIENT_LOCATION_LAT_COLUMN,
    CLIENT_LOCATION_LON_COLUMN,
    CLIENT_LOCATION_CITY_POP_COLUMN,
    MERCHANT_LOCATION_LAT_COLUMN,
    MERCHANT_LOCATION_LON_COLUMN,
    DISTANCE_FEATURE,
    CARD_TRANSACTION_COUNT_FEATURE,
    CARD_FRAUD_COUNT_FEATURE,
    CARD_FRAUD_RATE_FEATURE,
    CARD_AVG_AMOUNT_FEATURE,
]

CATEGORICAL_FEATURES = [
    MERCHANT_COLUMN,
    PAYMENT_CATEGORY_COLUMN,
    CLIENT_GENDER_COLUMN,
    CLIENT_LOCATION_STATE_COLUMN,
    CLIENT_JOB_COLUMN,
]

MODEL_TARGET_COLUMN = TARGET_COLUMN

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


def build_model() -> Pipeline:
    """Build the fraud detection preprocessing and Random Forest pipeline."""

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                categorical_transformer,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    **RANDOM_FOREST_PARAMS
                ),
            ),
        ]
    )

    return model