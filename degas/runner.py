import dataset
import model
import model.train
import click
import logging
import dotenv
import urllib.request
import tarfile
from pathlib import Path


@click.group()
def cli():
    pass


@cli.command()
@click.argument("url", type=click.STRING, default="https://s3.amazonaws.com/dga-data/dga_data_public.tar.gz")
@click.argument("output_filepath", default="data/raw")
def download_data(url, output_filepath: click.Path):
    """
    Downloads data from some location (s3, by default)

    This is just to avoid putting overly large data files in Git
    """
    click.echo('Downloading data from S3')
    filename, headers = urllib.request.urlretrieve(url)
    logging.info("Downloaded %s to %s", url, filename)
    with tarfile.open(filename) as tar:
        tar.extractall(str(output_filepath))
        logging.info("Extracted %s to %s", filename, output_filepath)


@cli.command()
@click.argument('input_filepath', type=click.Path(exists=True), default="data/raw")
@click.argument('output_filepath', type=click.Path(), default="data/processed")
def process_data(input_filepath: click.Path, output_filepath: click.Path) -> None:
    """
    Reads in raw data files (in various formats), does any necessary processing, and writes them into a single CSV file
    """
    click.echo('Processing raw data from {} and writing into {}'.format(input_filepath, output_filepath))
    dataset.process(str(input_filepath), str(output_filepath))


@cli.command()
@click.argument('input_filepath', type=click.Path(exists=True), default="data/processed")
@click.option("--epochs", default=100, show_default=True)
@click.option("--kfold_splits", default=None, show_default=True)
def train_model(input_filepath: click.Path, epochs: int = 100, kfold_splits: int = 3) -> None:
    """
    Reads processed datafile (as created by 'process_data') and trains a model on it.

    :param input_filepath: path to file or directory containing processed data
    :param epochs: Max epochs to train. Training will exit early if training is no longer progressing
    :param kfold_splits: by default, we will do only one train/test split. Set this to a number between 1 and to use
    kfold splits instead.
    """
    click.echo('Training the model')
    model.train.main(str(input_filepath), epochs, kfold_splits)


@cli.command()
@click.option("--port", default=8080, show_default=True)
def run_server(port: int = 8080):
    """
    Start a simple Flask webserver to serve the generated model

    Note that for a high-volume production site, consider using Tensorflow Serving (see the README). However, this is
    handy for development and testing.
    """
    click.echo("Starting server on port {}".format(port))
    pass


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    dotenv.load_dotenv(dotenv.find_dotenv())
    cli()
