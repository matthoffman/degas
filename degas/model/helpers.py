import numpy as np
from typing import List
from tensorflow.python.keras.preprocessing import sequence
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
    return [domain_name_dictionary.get(y, domain_name_dictionary.get(np.NaN)) for y in domain.lower()]


def prep_data(data, max_length):
    X = (data["domain"]
         .apply(lambda x: domain_to_ints(x))
         .pipe(sequence.pad_sequences, maxlen=max_length))
    y = data["class"]
    return X, y


