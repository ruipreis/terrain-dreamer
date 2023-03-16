# Separate the dataset into train and test
from typing import Tuple
import random
from pathlib import Path
import shutil


def split_train_test(
    samples: list, percentage_train: float = 0.9, limit: int = 20000
) -> Tuple[list, list]:
    random.shuffle(samples)
    samples = samples[:limit]
    split_index = int(len(samples) * percentage_train)
    return samples[:split_index], samples[split_index:]


def create_folders(
    train_folder: Path, test_folder: Path, train_samples: list, test_samples: list
):
    train_folder.mkdir(parents=True, exist_ok=True)
    test_folder.mkdir(parents=True, exist_ok=True)

    # Copy all the files to the train and test folders
    for sample in train_samples:
        shutil.copy(sample, train_folder / sample.name)

    for sample in test_samples:
        shutil.copy(sample, test_folder / sample.name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--percentage", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=20000)
    args = parser.parse_args()

    # Get all the samples
    train_samples, test_samples = split_train_test(
        list(args.dataset.glob("*.npz")), args.percentage, args.limit
    )

    # Create the folders
    create_folders(args.train, args.test, train_samples, test_samples)
