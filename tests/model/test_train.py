from ..context import degas
from pathlib import Path
import os
import pandas as pd


# TODO: I'm sure there are some utility methods to test here...


def test_build_model():
    model = degas.model.train.build_model()
    print("Model: {}".format(model))
    # TODO: what is worth asserting here?
