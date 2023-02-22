# -*- coding: utf-8 -*-
# TODO: The encoding statement required else errors are generated. Find a more elegant way to overcome errors.

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.features.build_features import FeatureEngineer
from src.data.load_data import DataLoader
from src.data.wrangle_data import DataWrangler
from src.models.train_model import ModelTrainer
from src.data.data_class import BatteryData

# Turn GPU support for Tensorflow off. To turn it on, comment these lines out.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """
    Model building pipeline providing processing for reading files from AWS Data Lake, wrangling data in Python and
    writing enriched data to ../processed directory or back to Data Lake.
    """
    logger = logging.getLogger(__name__)
    #    logger.info('Creating enriched data set from raw data read from AWS MySQL source.')
    logger.info('Creating modelling dataset for the Long Live Battery Project.')

    # Creating data loading and wrangling objects.
    # TODO: Need to look at factory method here for different model_id's used

    # Create data pipeline objects
    battery_data = BatteryData("long_live", None, None)
    data_loader = DataLoader("long_live", battery_data)
    data_wrangler = DataWrangler("long_live")
    feature_engineer = FeatureEngineer("long_live")

    # Load data from either SQL or pickle batch files depending on model i.e. battery_island (see constructor
    # arguments)
    data_loader.read_data_from_source()  # For long live these are 3 pkl batch files in external dir.
                                         # Output is stored in processed.pkl in interim dir.

    # Clean data, for 'long live' data this means...
    results = data_wrangler.clean_data(data_loader.get_dictionary())
    data_loader.save_preprocessed_data(results)

    # Reload new processed data (was also loaded during instantiation, but could have been changed).
    feature_engineer.load_processed_battery_data()
    split = feature_engineer.load_train_test_split()

    # Create tensorflow variables, this means...
    feature_engineer._write_to_tfrecords(train_test_split=split)

    # Create model trainer, this means...
    model_trainer = ModelTrainer(feature_engineer)

    # Train model, this means...
    model_trainer.train_and_evaluate()

    # Reading data into dataframe.


#    logger.info('Reading data from AWS MySQL database.')
#    df = data_loader.read_data_from_source(input_filepath)
#
# Wrangling data in dataframe.
#    logger.info('Wrangling battery island testing data.')
#    df = data_wrangler.process_data(df)
#
# Saving wrangled data in pickle file.
#    logger.info('Saving processed data.')
#    output_file = output_filepath + "/model_dataframe"
#    data_loader.write_data_to_pickle(df, output_file)
#

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # Find .env automagically by walking up directories until it's found, then
    # Load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
