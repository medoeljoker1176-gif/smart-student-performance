"""
train_model.py - entry point kept from the first version.
Trains, tunes and compares the models, then saves models/, reports/.

Run:  python train_model.py                  (synthetic dataset, the default)
      python train_model.py --dataset uci    (real UCI data; run `python prepare_uci.py` first)
"""
import argparse

from src.config import DATASETS, DEFAULT_DATASET
from src.model_training import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the student-performance models.")
    parser.add_argument("--dataset", choices=list(DATASETS), default=DEFAULT_DATASET)
    main(dataset=parser.parse_args().dataset)
