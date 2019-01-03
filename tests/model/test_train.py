from ..context import degas
from tensorflow.python.keras.models import Model


# TODO: I'm sure there are some utility methods to test here...


def test_build_model():
    model: Model = degas.model.train.build_model()
    # this validates that we can build and compile it w/o error, which catches the most common issues in model creation
    print("Model: {}".format(model))


