import tensorflow as tf
import numpy as np
import os
from .train import as_keras_metric
from tensorflow.python.keras.models import Model
from .helpers import prep_data


def load_model() -> Model:
    precision = as_keras_metric(tf.metrics.precision)
    recall = as_keras_metric(tf.metrics.recall)
    model_path = os.path.join("models", "degas", "nyu_model.h5")
    model = tf.keras.models.load_model(model_path, custom_objects={"precision": precision, "recall": recall})
    print("Loaded model {}".format(model))
    return model


def predict(model: Model, domains: np.ndarray) -> np.ndarray:
    """
    Given a list of domains as input, returns a list of booleans, where True means it is predicted to be a DGA, and
    false means it is predicted to be benign
    """
    predictions = model.predict_on_batch(prep_data(domains))
    return predictions
