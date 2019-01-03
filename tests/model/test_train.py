from ..context import degas
from pathlib import Path
import os
import pandas as pd
import numpy as np
import sklearn
from tensorflow.python.keras.models import Model


# TODO: I'm sure there are some utility methods to test here...


def test_build_model():
    model: Model = degas.model.train.build_model()
    # this validates that we can build and compile it w/o error, which catches the most common issues in model creation
    print("Model: {}".format(model))


def test_print_metrics():
    y_true = [0, 1, 0, 1]
    y_pred = [1, 1, 1, 0]

    confusion_matrix, precision, recall, f1 = degas.model.train.print_metrics(y_true, y_pred)
    # figure out what they should be
    expected_conf_matrix: np.ndarray = sklearn.metrics.confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0])
    assert np.all(confusion_matrix == expected_conf_matrix)
    exp_precision, exp_recall, exp_f1, exp_support = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred)
    print("Precision: {} (expected {}), recall: {} (expected {}), f1: {} (expected {})".format(
        precision, exp_precision[-1], recall, exp_recall[1], f1, exp_f1))
    assert precision == exp_precision[-1]
    assert recall == exp_recall[-1]
    assert f1 == exp_f1[-1]


