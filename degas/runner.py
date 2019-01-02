import dataset
import model.train
import click
import logging
import dotenv
from pathlib import Path



@click.group()
def cli():
    pass


@cli.command()
def download_data():
    click.echo('Downloading data from S3')
    click.echo("(not yet implemented :( ")


@cli.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def process_data(input_filepath: click.Path, output_filepath: click.Path) -> None:
    click.echo('Processing raw data from {} and writing into {}'.format(input_filepath, output_filepath))
    dataset.process(str(input_filepath), str(output_filepath))


@cli.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.option("--epochs", default=100, show_default=True)
@click.option("--kfold_splits", default=None, show_default=True)
def train_model(input_filepath: click.Path, epochs: int = 100, kfold_splits: int = 3) -> None:
    click.echo('Training the model')
    model.train.main(str(input_filepath), epochs, kfold_splits)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    dotenv.load_dotenv(dotenv.find_dotenv())
    cli()
