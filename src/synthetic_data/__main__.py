import argparse
import sys
from pathlib import Path

from .generator import SyntheticData


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data")
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        required=True,
        help="Path to the synthetic data config",
    )
    args = parser.parse_args()

    # Check if the path and the files exist
    path = Path(args.path)
    if not path.exists():
        print(f"The path {args.path} does not exist")
        sys.exit(1)
    if not (path / "columns.json").exists():
        print(f"The file {path / 'columns.json'} does not exist")
        sys.exit(1)
    if not (path / "data.json").exists():
        print(f"The file {path / 'data.json'} does not exist")
        sys.exit(1)

    # Initialize and use your BaseSyntheticData class
    synthetic_data = SyntheticData(args.path)
    synthetic_data.generate_datasets()
    synthetic_data.export_datasets()


if __name__ == "__main__":
    main()
