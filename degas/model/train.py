import logging
from typing import List

import os
import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.python.keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, History)
from tensorflow.python.keras.layers import (Conv1D, MaxPooling1D, ThresholdedReLU, Embedding,)
from tensorflow.python.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.python.keras.models import Model
from sklearn.model_selection import train_test_split, StratifiedKFold

from .helpers import prep_dataframe, prep_data, print_metrics, as_keras_metric

# TODO: temporarily commenting this out, but worth revisiting to remove the magic strings
# from ..dataset import DATA_KEY, LABEL_KEY, DATASET_FILENAME

logger = logging.getLogger(__name__)


def build_model() -> Model:
    """
    This is the code for the "NYU" model from "Character Level Based Detection of DGA Domain Names"
    (http://faculty.washington.edu/mdecock/papers/byu2018a.pdf)
    which is itself adapted from X. Zhang, J. Zhao, and Y. LeCun, “Character-level convolutional networks for text
    classification,” in Advances in Neural Information Processing Systems, vol. 28, 2015, pp. 649–657.
    """
    # a constant here representing the maximum expected length of a domain.
    max_length = 75
    main_input = Input(shape=(max_length,), dtype="int32", name="main_input")
    embedding = Embedding(input_dim=128, output_dim=128, input_length=max_length)(main_input)
    conv1 = Conv1D(filters=128, kernel_size=3, padding="same", strides=1)(embedding)
    thresh1 = ThresholdedReLU(1e-6)(conv1)
    max_pool1 = MaxPooling1D(pool_size=2, padding="same")(thresh1)
    conv2 = Conv1D(filters=128, kernel_size=2, padding="same", strides=1)(max_pool1)
    thresh2 = ThresholdedReLU(1e-6)(conv2)
    max_pool2 = MaxPooling1D(pool_size=2, padding="same")(thresh2)
    flatten = Flatten()(max_pool2)
    fc = Dense(64)(flatten)
    thresh_fc = ThresholdedReLU(1e-6)(fc)
    drop = Dropout(0.5)(thresh_fc)
    output = Dense(1, activation="sigmoid")(drop)
    model = Model(inputs=main_input, outputs=output)
    precision = as_keras_metric(tf.metrics.precision)
    recall = as_keras_metric(tf.metrics.recall)
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["mae", "mean_squared_error", "acc", precision, recall],
    )
    return model


def get_callbacks(model_filename, patience_stopping=5, patience_lr=10):
    early_stopping = EarlyStopping(monitor="val_loss", patience=patience_stopping, verbose=1)
    mcp_save = ModelCheckpoint(model_filename, save_best_only=True, monitor="val_loss", mode="min")
    reduce_lr_loss = ReduceLROnPlateau(
        monitor="loss",
        factor=0.1,
        patience=patience_lr,
        verbose=1,
        epsilon=1e-4,
        mode="min",
    )
    return [early_stopping, mcp_save, reduce_lr_loss]


def fit_and_evaluate(model: Model, model_filename: str, t_x, val_x, t_y, val_y, epochs=20, batch_size=128) -> History:
    results = model.fit(
        t_x,
        t_y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(model_filename),
        verbose=1,
        validation_data=[val_x, val_y],
    )
    logging.info("Score against validation set: %s", model.evaluate(val_x, val_y))
    return results


def export_model(model: Model):
    # Ignore dropout at inference
    tf.keras.backend.set_learning_phase(0)
    export_path = os.path.join(
        "models", "degas", "1"
    )  # TODO: how/when do we want to increment versions?
    with tf.keras.backend.get_session() as sess:
        tf.saved_model.simple_save(
            sess,
            export_path,
            inputs={"input_image": model.input},
            outputs={t.name: t for t in model.outputs},
        )


def run(data: pd.DataFrame, num_epochs=100, batch_size=256, max_length=75) -> History:
    # Convert characters to lists of ints and pad them
    logger.info("Preparing data")
    X: np.ndarray = prep_data(data["domain"], max_length)
    y: np.ndarray = data["class"]
    t_x, val_x, t_y, val_y = train_test_split(X, y, test_size=0.2)
    logger.info("created a training set of %i records (of which %.2f%% are DGAs) and a test set of %i records "
                "(of which %.2f%% are DGAs)", len(t_x), t_y.mean() * 100, len(val_x), val_y.mean() * 100)

    logger.info("Building model")
    model: Model = build_model()

    logger.info("Starting training")
    model_filename = os.path.join("models", "degas", "nyu_model.h5")
    history: History = fit_and_evaluate(model, model_filename, t_x, val_x, t_y, val_y, num_epochs, batch_size)
    logger.info("Last training history: " + str(history))
    logger.info("Exporting model for serving")
    export_model(model)

    predict_y = model.predict(val_x, batch_size=batch_size, verbose=1)
    print_metrics(val_y, predict_y)

    return history


def run_kfold(data: pd.DataFrame, num_epochs=100, kfold_splits=2, batch_size=256, max_length=75) -> List[History]:
    """
    Variant of run that uses Stratified KFold.
    Not currently used (kfold isn't necessary with this dataset) but kept around for the moment in case it's useful.
    """
    logger = logging.getLogger(__name__)

    logger.info("Preparing dataset")
    # Convert characters to lists of ints and pad them
    X, y = prep_dataframe(data, max_length)
    # TODO: pull out a holdout set (test set) to test against after kfold?

    # save the model history in a list after fitting so that we can plot later
    model_history = []
    skf = StratifiedKFold(n_splits=kfold_splits)
    for i, (train_indices, val_indices) in enumerate(skf.split(X, y)):
        logger.info("Starting fold %u out of %u", i + 1, kfold_splits)
        t_x, val_x = X[train_indices], X[val_indices]
        t_y, val_y = y[train_indices], y[val_indices]

        model: Model = build_model()
        weights_filename = "nyu_model_fold" + str(i) + "_weights.h5"
        history: History = fit_and_evaluate(model, weights_filename, t_x, val_x, t_y, val_y, num_epochs, batch_size)
        model_history.append(history)
        accuracy_history = history.history["acc"]
        # val_accuracy_history = history.history['val_acc']
        logger.info("Last training accuracy: " + str(accuracy_history[-1]))

    return model_history


def main(input_filepath: str, epochs: int = 100, kfold_splits: int = 3) -> None:
    logging.info("load up some data")
    input_path = Path(input_filepath)
    # if the input was a directory, add our default filename.
    # otherwise, we'll assume it was an explicit file argument
    if input_path.is_dir():
        input_path = input_path.joinpath("dataset.csv.gz")

    data: pd.DataFrame = pd.read_csv(input_path)
    logging.info("Starting model training")
    if kfold_splits is not None:
        run_kfold(data, num_epochs=epochs, kfold_splits=kfold_splits)
    else:
        run(data, num_epochs=epochs)
