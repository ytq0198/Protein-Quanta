"""Print a compact schema summary for one MISATO train sample."""

import argparse

from NeuralMD.datasets.MISATO import DatasetMISATOSemiFlexibleMultiTrajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    dataset = DatasetMISATOSemiFlexibleMultiTrajectory(args.data_root, mode="train")
    sample = dataset[0]
    print("length", len(dataset))
    for key in sample.keys():
        value = getattr(sample, key)
        print(key, getattr(value, "shape", None), getattr(value, "dtype", None))


if __name__ == "__main__":
    main()
