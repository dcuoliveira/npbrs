import os
import argparse
import pickle

from settings import INPUT_PATH
from data.ETFsLoader import ETFsLoader

parser = argparse.ArgumentParser()

parser.add_argument('--strategy_name', type=str, help='strategy name to generate directory.', default="training_etfstsm")

if __name__ == "__main__":

    args = parser.parse_args()

    strategy_name = args.strategy_name

    etfs_loader = ETFsLoader()
    bars = etfs_loader.bars

    # check if repo exists
    if not os.path.exists(os.path.join(INPUT_PATH, strategy_name)):
        os.makedirs(os.path.join(INPUT_PATH, strategy_name))

    results = {"bars": bars}

    with open(os.path.join(INPUT_PATH, strategy_name, f"{strategy_name}.pickle"), 'wb') as handle:
        pickle.dump(results,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL)