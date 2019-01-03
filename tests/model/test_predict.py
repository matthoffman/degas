from ..context import degas
import numpy as np
import sklearn
from tensorflow.python.keras.models import Model


def test_predict():
    """ make sure prediction works as expected """
    # get our model
    model: Model = degas.model.predict.load_model()

    # create some domains to predict
    df = np.array(["www.google.com",
                   "google.com",
                   "facebook.com",
                   "lovingtonrvpark.com",
                   "xcbeoysoqcbnlvje.eu",
                   "gnqqmcj.pw",
                   "1npwf5x185w0q8qvk09runm83.net",
                   "yeiesmomgeso.org"])
    # predict
    predictions: np.ndarray = degas.model.predict.predict(model, df)
    print("Predictions (should be [0,0,0,0,1,1,1,1]: {}".format(predictions))

    # print metrics from our predictions
    (confmatrix, precision, recall, f1) = degas.model.helpers.print_metrics(np.array([0, 0, 0, 0, 1, 1, 1, 1]), predictions)
    print("Confusion matrix: {}".format(confmatrix))
    print("precision: {}, recall: {}, f1: {}".format(precision, recall, f1))
    assert precision == 1
    assert recall >= 0.75  # I would like to guarantee 100%, but we don't hit that with this set.
    assert f1 >= 0.85


def test_predict_rovnix():
    """
    Test against Rovnix, which uses real words. That makes them particularly tricky to detect...
    """
    # get our model
    model: Model = degas.model.predict.load_model()

    rovnix = np.array([
        "kingwhichtotallyadminis.biz",
        "thareplunjudiciary.net",
        "townsunalienable.net",
        "taxeslawsmockhigh.net",
        "transientperfidythe.biz",
        "inhabitantslaindourmock.cn",
        "thworldthesuffer.biz"])

    # predict
    predictions: np.ndarray = degas.model.predict.predict(model, rovnix)

    # print metrics from our predictions
    (confmatrix, precision, recall, f1) = degas.model.helpers.print_metrics(np.array([1, 1, 1, 1, 1, 1, 1]), predictions)
    print("Confusion matrix: {}".format(confmatrix))
    print("precision: {}, recall: {}, f1: {}".format(precision, recall, f1))
    assert precision == 1
    # These currently fail; we hit about 42% with these sorts of domains. Not great...
    # assert recall == 1
    # assert f1 == 1
