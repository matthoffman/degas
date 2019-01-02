import logging
from typing import List

import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, History
from tensorflow.keras.layers import Conv1D, MaxPooling1D, ThresholdedReLU, Embedding
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import sequence
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold

# TODO: temporarily commenting this out, but worth revisiting to remove the magic strings
# from ..dataset import DATA_KEY, LABEL_KEY, DATASET_FILENAME

# See: https://github.com/Yuren-Zhong/DeepDGA/blob/master/train.py

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
domain_name_dictionary = {'0': 0,
                          '1': 1,
                          '2': 2,
                          '3': 3,
                          '4': 4,
                          '5': 5,
                          '6': 6,
                          '7': 7,
                          '8': 8,
                          '9': 9,
                          ':': 10,
                          '-': 11,
                          '.': 12,
                          '/': 13,
                          '_': 14,
                          'a': 15,
                          'b': 16,
                          'c': 17,
                          'd': 18,
                          'e': 19,
                          'f': 20,
                          'g': 21,
                          'h': 22,
                          'i': 23,
                          'j': 24,
                          'k': 25,
                          'l': 26,
                          'm': 27,
                          'n': 28,
                          'o': 29,
                          'p': 30,
                          'q': 31,
                          'r': 32,
                          's': 33,
                          't': 34,
                          'u': 35,
                          'v': 36,
                          'w': 37,
                          'x': 38,
                          'y': 39,
                          'z': 40,
                          np.NaN: 41}


def build_model() -> Model:
    """
    This is the code for the "NYU" model from "Character Level Based Detection of DGA Domain Names"
    (http://faculty.washington.edu/mdecock/papers/byu2018a.pdf)
    which is itself adapted from X. Zhang, J. Zhao, and Y. LeCun, “Character-level convolutional networks for text
    classification,” in Advances in Neural Information Processing Systems, vol. 28, 2015, pp. 649–657.
    """
    # a constant here representing the maximum expected length of a domain.
    max_length = 75
    main_input = Input(shape=(max_length,), dtype='int32', name='main_input')
    embedding = Embedding(input_dim=128, output_dim=128,
                          input_length=max_length)(main_input)
    conv1 = Conv1D(filters=128, kernel_size=3, padding='same', strides=1)(embedding)
    thresh1 = ThresholdedReLU(1e-6)(conv1)
    max_pool1 = MaxPooling1D(pool_size=2, padding='same')(thresh1)
    conv2 = Conv1D(filters=128, kernel_size=2, padding='same', strides=1)(max_pool1)
    thresh2 = ThresholdedReLU(1e-6)(conv2)
    max_pool2 = MaxPooling1D(pool_size=2, padding='same')(thresh2)
    flatten = Flatten()(max_pool2)
    fc = Dense(64)(flatten)
    thresh_fc = ThresholdedReLU(1e-6)(fc)
    drop = Dropout(0.5)(thresh_fc)
    output = Dense(1, activation='sigmoid')(drop)
    model = Model(inputs=main_input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mae', 'mean_squared_error', 'acc'])
    return model


def domain_to_ints(domain: str) -> List[int]:
    """
    Converts the given domain into a list of ints, given the static dictionary defined above.
    Converts the domain to lower case, and uses a set value (mapped to np.NaN) for unknown characters.
    """
    return [domain_name_dictionary.get(y, domain_name_dictionary.get(np.NaN)) for y in domain.lower()]


def get_callbacks(weights_filename, patience_stopping=5, patience_lr=10):
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience_stopping, verbose=1)
    mcp_save = ModelCheckpoint(weights_filename, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4,
                                       mode='min')
    return [early_stopping, mcp_save, reduce_lr_loss]


def fit_and_evaluate(model: Model, weights_filename: str, t_x, val_x, t_y, val_y, epochs=20, batch_size=128) -> History:
    results = model.fit(t_x, t_y, epochs=epochs, batch_size=batch_size, callbacks=get_callbacks(weights_filename),
                        verbose=1, validation_data=[val_x, val_y])
    logging.info("Score against validation set: %s", model.evaluate(val_x, val_y))
    return results


def export_model(model: Model):
    # Ignore dropout at inference
    tf.keras.backend.set_learning_phase(0)
    export_path = 'models/degas/1'
    with tf.keras.backend.get_session() as sess:
        tf.saved_model.simple_save(
            sess,
            export_path,
            inputs={'input_image': model.input},
            outputs={t.name: t for t in model.outputs})


def run(data: pd.DataFrame, num_epochs=100, batch_size=256, max_length=75) -> History:
    logger = logging.getLogger("train")
    # Convert characters to lists of ints and pad them
    logger.info("Preparing data")
    X, y = prep_data(data, max_length)
    t_x, val_x, t_y, val_y = train_test_split(X, y, test_size=0.2)
    logger.info("created a training set of %i records (of which %.2f%% are DGAs) and a test set of %i records "
                "(of which %.2f%% are DGAs)", len(t_x), t_y.mean() * 100, len(val_x), val_y.mean() * 100)

    logger.info("Building model")
    model: Model = build_model()

    logger.info("Starting training")
    weights_filename = "nyu_model_weights.h5"
    history: History = fit_and_evaluate(model, weights_filename, t_x, val_x, t_y, val_y, num_epochs, batch_size)
    logger.info("Last training history: " + str(history))

    export_model(model)

    return history


def run_kfold(data: pd.DataFrame, num_epochs=100, kfold_splits=2, batch_size=256, max_length=75) -> List[History]:
    """
    Variant of run that uses Stratified KFold
    """
    logger = logging.getLogger(__name__)

    logger.info("Preparing dataset")
    # Convert characters to lists of ints and pad them
    X, y = prep_data(data, max_length)
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
        accuracy_history = history.history['acc']
        # val_accuracy_history = history.history['val_acc']
        logger.info("Last training accuracy: " + str(accuracy_history[-1]))
        # t_probs = model.predict_on_batch(holdout["domain_ints"])
        # t_auc = roc_auc_score(holdout[LABEL_KEY], t_probs)
        #
        # logger.info('Epoch %d: auc = %f (best=%f)', ep, t_auc, best_auc)
        #
        # if t_auc > best_auc:
        #     best_auc = t_auc
        #     best_iter = ep
        #
        #     probs = model.predict_on_batch(test["domain_ints"])
        #     confmatrix = confusion_matrix(test[DATA_KEY], probs > .5)
        #     out_data = {'y': test[DATA_KEY], 'labels': test[LABEL_KEY], 'probs': probs, 'epochs': ep,
        #                 'confusion_matrix': confmatrix}
        #
        #     logger.info("Confusion matrix: %s", confmatrix)

    return model_history


def prep_data(data, max_length):
    X = (data["domain"]
         .apply(lambda x: domain_to_ints(x))
         .pipe(sequence.pad_sequences, maxlen=max_length))
    y = data["class"]
    return X, y


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


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
