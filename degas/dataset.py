# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from glob import glob
import pandas as pd
from typing import List
from csv import QUOTE_ALL
# from sklearn.model_selection import train_test_split


#####################
# some constants
#####################

# key for our dataframe that contains the data (the domains)
DATA_KEY = 'domain'
# key for our dataframe that contains the labels (whether it's a DGA or not)
LABEL_KEY = 'class'
DATASET_FILENAME = "dataset.csv.gz"


def process(input_filepath: str, output_filepath: str) -> None:
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    logger.info("Loading DGA datasets")
    dgas = concat([load_andrewaeva(join(input_filepath, "andrewaeva-dga-list.csv.gz")),
                   load_bambenek(join(input_filepath, "bambenek.*.csv.gz"))],
                  dga=True)

    logger.info("Loading benign domain datasets")
    benign = concat([load_cisco(join(input_filepath, "cisco-umbrella-top-1m.csv.zip")),
                     load_majestic_million(join(input_filepath, "majestic_million.csv.gz")),
                     load_alexa(join(input_filepath, "alexa.csv.gz")),
                     load_top10million(join(input_filepath, "top10milliondomains.csv.zip"))],
                    dga=False)

    logger.info("Loaded a total of %i DGA domains and %i benign domains", len(dgas), len(benign))
    logger.info("There are %i *unique* benign domains", len(benign.drop_duplicates()))
    full: pd.DataFrame = pd.concat([dgas, benign], ignore_index=True)
    logger.info("created a dataset of %i records (of which %.2f%% are DGAs)", len(full), full[LABEL_KEY].mean()*100)
    full.to_csv(join(output_filepath, DATASET_FILENAME), header=True, index=False, compression="gzip")
    logger.info("dataset creation complete. dataset.csv.gz written to %s", output_filepath)

    # train, test = train_test_split(full, test_size=0.2)
    # train: pd.DataFrame

    # logger.info("created a training set of %i records (of which %.2f%% are DGAs) and a test set of %i records "
    #      "(of which %.2f%% are DGAs)", len(train), train[LABEL_KEY].mean()*100, len(test), test[LABEL_KEY].mean()*100)
    # train.to_csv(join(output_filepath, "train.csv.gz"), header=True, index=False, compression="gzip")
    # test.to_csv(join(output_filepath, "test.csv.gz"), header=True, index=False, compression="gzip")
    # logger.info("dataset creation complete. test.csv and train.csv written to %s", output_filepath)


def concat(dataframes: List[pd.DataFrame], dga=False) -> pd.DataFrame:
    df = pd.concat(dataframes, ignore_index=True)
    if dga:
        df[LABEL_KEY] = 1
    else:
        df[LABEL_KEY] = 0
    return df


def load_top10million(path: Path) -> pd.DataFrame:
    """
    A list of the top 10 million domains according to the Open PageRank Initiative, based on Common Crawl data.
    Since these are actually crawled, we assume they are not DGAs.

    This file has a header, and looks like this:
    "Rank","Domain","Open Page Rank"
    "1","facebook.com","10.00"
    """
    logging.info(" - reading %s", path)
    df = pd.read_csv(path, names=[DATA_KEY], usecols=[1], skiprows=1, quoting=QUOTE_ALL)
    logging.info(" - read %i records from %s", len(df), path)
    return df


def load_majestic_million(path: Path) -> pd.DataFrame:
    """
    load the "majestic million" top 1 million website dataset. This has significant overlap with Alexa and Cisco's "top
    1 million domain" datasets, obviously, but that's OK. No harm in that.

    This actually has a header row. First couple lines are:
    GlobalRank,TldRank,Domain,TLD,RefSubNets,RefIPs,IDN_Domain,IDN_TLD,PrevGlobalRank,PrevTldRank,PrevRefSubNets,PrevRefIPs
1,1,google.com,com,487267,3086039,google.com,com,1,1,487043,3085865
    """
    logging.info(" - reading %s", path)
    df = pd.read_csv(path, names=[DATA_KEY], usecols=[2], skiprows=1)
    logging.info(" - read %i records from %s", len(df), path)
    return df


def load_subset(path: Path) -> pd.DataFrame:
    """ A small subset of the input data, for testing purposes.
    """
    # chr(1) is ctrl-A, which is a pretty vile separator char, TBH. I mean, couldn't it at least be ctrl-^
    # ("record separator") or ctrl-_ ("unit separator")?
    logging.info(" - reading %s", path)
    subset = pd.read_csv(path, delimiter=chr(1), names=[DATA_KEY, "desc", "class"], usecols=[0, 2], header=None,
                         error_bad_lines=False)
    logging.info(" - read %i records from %s", len(subset), path)
    return subset


def load_andrewaeva(path: Path) -> pd.DataFrame:
    """ This dataset is from andrewaeva (github.com:andrewaeva/DGA.git), where it is used as the training set of for a
        couple of DGA detection models.
        It's a simple CSV dataset without a header, with 2 columns: domain, and a number representing the algorithm that
        generated it.
        For our purposes, we don't care which algorithm generated the domain, so we'll just pull the first column.
    """
    logging.info(" - reading %s", path)
    dga1 = pd.read_csv(path, delimiter="\\s+", header=None, names=[DATA_KEY, "dga_type"], usecols=[0])
    logging.info(" - read %i records from %s", len(dga1), path)
    return dga1


def load_bambenek(path: Path) -> pd.DataFrame:
    """ Bambenek consulting publishes a regular feed of DGA-generated URLs to [TODO: URL]

    It has a license info in a header, but not an actual header line.
    The actual file content looks like this:
    plvklpgwivery.com,Domain used by Cryptolocker - Flashback DGA for 23 Jun 2018,2018-06-23,http://osint.bambenekconsulting.com/manual/cl.txt

    There are several files, currently, so this handles glob-style wildcards in the incoming path
    """
    dga = pd.DataFrame()
    for p in glob(str(path)):
        logging.info(" - reading %s", p)
        this_dga = pd.read_csv(p, header=None, comment="#", names=[DATA_KEY], usecols=[0])
        logging.info(" - read %i records from %s", len(this_dga), path)
        dga = dga.append(this_dga, ignore_index=True, verify_integrity=True)
    dga.drop_duplicates(inplace=True)
    return dga


def load_cisco(path: Path) -> pd.DataFrame:
    """ Cisco publishes a "top 1 million URLs" dataset. Being popularity-based, we assume that none of these are DGAs.
    """
    logging.info(" - reading %s", path)
    benign = pd.read_csv(path, header=None, comment="#", names=["rank", DATA_KEY], usecols=[1])
    logging.info(" - read %i records from %s", len(benign), path)
    return benign


def load_alexa(path: Path) -> pd.DataFrame:
    """
    Load the top 1 million websites according to Alexa.
    This is a space-separated file with 2 columns, domain and rank (I believe?). We only care about the domain.
    """
    logging.info(" - reading %s", path)
    df = pd.read_csv(path, header=None, delimiter="\\s+", names=[DATA_KEY], usecols=[0])
    logging.info(" - read %i records from %s", len(df), path)
    return df


def join(base: str, filename: str) -> Path:
    """
    Yes, we could use os.path.join, but for reasons I no longer remember I'm passing around Path objects instead.
    Maybe I intended to use some of the cleverness that Path allows?
    """
    return Path(base).joinpath(filename)
