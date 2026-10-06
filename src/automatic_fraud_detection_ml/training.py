import pandas as pd

from sklearn.pipeline import Pipeline


def train_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Train the fraud detection model. Pipeline include preprocessing """

    model.fit(X_train, y_train)

    return model