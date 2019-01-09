import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from tensorflow.python.keras.preprocessing import sequence


logger = logging.getLogger(__name__)

# TODO: temporarily commenting this out, but worth revisiting to remove the magic strings
# from ..dataset import DATA_KEY, LABEL_KEY, DATASET_FILENAME

# Static dictionary of allowed characters in domains.
# Not elegant, but does the job. Could create it algorithmically using ord() or whatever, but at the end of the day this
# is easier to grok.
# Note that I *thought* that the new TLDs introduced a much wider assortment of allowed characters, but this dictionary
# was built from the actual characters pulled from our test set. May be something to revisit, though.
# To reconstruct:
# dict = {chr(i):i-48 for i in range(48,59)}
# dict.update({chr(i):len(dict)+i-45 for i in range(45,48)})
# dict['_'] = len(dict)
# dict.update({chr(i):len(dict)+i-97 for i in range(97,123)})
domain_name_dictionary = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, ':': 10,
                          '-': 11, '.': 12, '/': 13, '_': 14, 'a': 15, 'b': 16, 'c': 17, 'd': 18, 'e': 19, 'f': 20,
                          'g': 21, 'h': 22, 'i': 23, 'j': 24, 'k': 25, 'l': 26, 'm': 27, 'n': 28, 'o': 29, 'p': 30,
                          'q': 31, 'r': 32, 's': 33, 't': 34, 'u': 35, 'v': 36, 'w': 37, 'x': 38, 'y': 39, 'z': 40,
                          np.NaN: 41}


def domain_to_ints(domain: str) -> List[int]:
    """
    Converts the given domain into a list of ints, given the static dictionary defined above.
    Converts the domain to lower case, and uses a set value (mapped to np.NaN) for unknown characters.
    """
    return [
        domain_name_dictionary.get(y, domain_name_dictionary.get(np.NaN))
        for y in domain.lower()
    ]


# TODO: put the max_length constant somewhere
def prep_dataframe(data: pd.DataFrame, max_length=75) -> Tuple[np.ndarray, np.ndarray]:
    X = (data["domain"]
         .apply(lambda x: domain_to_ints(x))
         .pipe(sequence.pad_sequences, maxlen=max_length))
    y = data["class"]
    return X, y


def prep_data(data: np.ndarray, max_length=75) -> np.ndarray:
    """ TODO: DRY; combine this with prep_dataframe above"""
    return sequence.pad_sequences(
        np.array([domain_to_ints(x) for x in data]),
        maxlen=max_length)


def as_keras_metric(method):
    """ from https://stackoverflow.com/questions/43076609/how-to-calculate-precision-and-recall-in-keras """
    import functools

    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        tf.keras.backend.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value

    return wrapper


def print_metrics(val_y: np.ndarray, predict_y: np.ndarray):
    confmatrix: np.ndarray = sklearn.metrics.confusion_matrix(val_y, predict_y > 0.5)

    tn, fp, fn, tp = confmatrix.ravel()
    num_pred_positives = tp + fp
    num_positives = tp + fn
    num_negatives = tn + fp
    precision = tp / num_pred_positives
    recall = tp / num_positives
    fpr = fp / num_negatives
    fnr = fn / num_positives
    # sklearn could calculate this for us as well
    f1 = 2 * (precision * recall) / (precision + recall)
    logger.info("precision: %.2f, TPR (recall): %.2f, FPR: %.2f, FNR (miss rate): %.2f. f1 score: %.2f",
                precision, recall, fpr, fnr, f1)
    print("precision: {}, TPR (recall): {}, FPR: {}, FNR (miss rate): {}. f1 score: {}".format(
        precision, recall, fpr, fnr, f1))

    # or, just let sklearn do it for us, and even print a pretty table :)
    report = sklearn.metrics.classification_report(
        val_y, predict_y > 0.5, labels=[0, 1], target_names=["benign", "DGA"]
    )

    logger.info(report)
    print(report)
    return confmatrix, precision, recall, f1
