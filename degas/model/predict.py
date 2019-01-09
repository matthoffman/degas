import tensorflow as tf
import numpy as np
import os
from typing import List
from .train import as_keras_metric
from tensorflow.python.keras.models import Model
from .helpers import prep_data


def model_versions() -> List[int]:
    """
    Get all known model versions.

    This is implemented as a simple listing of subdirectories of "models/degas" within the source.
    Note that Tensorflow Serving restricts these to be numbers, but this currently does not.

    :return: a list of version names which can be passed into load_model
    """
    with os.scandir(model_base_path()) as entries:
        return [entry.name for entry in entries if entry.is_dir]


def load_model(version=1) -> Model:
    """
    Load a model. Optionally takes the version # to load, or the latest if not specified
    :param version: defaults to 1, must be a valid directory under "models/degas/" containing a model definition.
    :return: a loaded model
    """
    precision = as_keras_metric(tf.metrics.precision)
    recall = as_keras_metric(tf.metrics.recall)
    model_path = os.path.join("models", "degas", str(version), "nyu_model.h5")
    model = tf.keras.models.load_model(model_path, custom_objects={"precision": precision, "recall": recall})
    print("Loaded model {} version {}".format(model, version))
    return model


def predict(model: Model, domains: np.ndarray) -> np.ndarray:
    """
    Given a list of domains as input, returns a list of booleans, where True means it is predicted to be a DGA, and
    false means it is predicted to be benign
    """
    predictions = model.predict_on_batch(prep_data(domains))
    return predictions


def model_base_path() -> str:
    return os.path.join("models", "degas")


