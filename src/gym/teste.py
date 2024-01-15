import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.conn_data import load_pickle, save_strat_opt_results
from settings import INPUT_PATH, OUTPUT_PATH

inputs = load_pickle(os.path.join(INPUT_PATH, "training_etfstsm", "training_etfstsm.pickle"))
