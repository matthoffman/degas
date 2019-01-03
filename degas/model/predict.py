import tensorflow as tf
import numpy as np
import os
from tensorflow.python.keras.models import Model
from typing import List


def load_model() -> Model:
    model= tf.keras.models.load_model(os.path.join('models', 'degas', 'nyu_model.h5'))
    print("Loaded model {}".format(model))
    return model


def predict(model: Model, domains: np.ndarray) -> np.ndarray:
    """
    Given a list of domains as input, returns a list of booleans, where True means it is predicted to be a DGA, and
    false means it is predicted to be benign
    """
    predictions = model.predict_on_batch(domains)
    return predictions

