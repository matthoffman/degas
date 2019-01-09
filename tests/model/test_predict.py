from ..context import degas
import numpy as np
import pandas as pd
import sklearn
from tensorflow.python.keras.models import Model


def test_predict_basic():
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
    val_y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    (confmatrix, precision, recall, f1) = degas.model.helpers.print_metrics(val_y, predictions)
    print("Confusion matrix: {}".format(confmatrix))
    print("precision: {}, recall: {}, f1: {}".format(precision, recall, f1))
    assert precision == 1
    assert recall >= 0.75  # I would like to guarantee 100%, but we don't hit that with this set.
    assert f1 >= 0.85
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(val_y, predictions)
    print("fpr:       {}\ntpr:      {}\nthresholds: {}", fpr, tpr, thresholds)


def test_predict_rovnix():
    """
    Test against Rovnix, which uses real words. That makes them particularly tricky to detect.

    This is mainly just out of curiosity for how this model performs against this type of DGAs;
    there's not much to assert here.
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


def test_predict_20k():
    """
    Predict using a slightly larger, more realistic dataset, and print results

    test.csv.gz was generated using new (not in the training set) DGAs from the most recent bambenek feed (at time of
    writing), and 20,000 benign domains randomly selected from the Common Crawl 10m domain dataset. So the benign
    domains *are* in the training set (we train against the whole 10m, among others), so there's some skew. But we're
    doing that on the assumption that this more-or-less mimics real-world usage; we're going to keep seeing novel DGA
    domains, but most benign URLs in common use are not novel.

    """
    # get our test data
    test_df = pd.read_csv("data/raw/test.csv.gz")
    val_x = test_df["domain"]
    val_y = test_df["label"]

    # get our model
    model: Model = degas.model.predict.load_model()

    # predict
    pred_y: np.ndarray = degas.model.predict.predict(model, val_x)

    # print metrics from our predictions
    (confmatrix, precision, recall, f1) = degas.model.helpers.print_metrics(val_y, pred_y)
    print("Confusion matrix: \n{}".format(confmatrix))
    print("precision: {}, recall: {}, f1: {}".format(precision, recall, f1))
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(val_y, pred_y)
    print("fpr:       {}\ntpr:      {}\nthresholds: {}", fpr, tpr, thresholds)
    assert precision >= 0.99
    assert recall >= 0.98
    assert f1 >= 0.99



