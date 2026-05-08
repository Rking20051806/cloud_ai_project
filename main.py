from __future__ import annotations

import argparse
from pathlib import Path

from scripts.inspect_dataset import main as inspect_dataset_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cloud AI Project")
    parser.add_argument("--inspect-data", action="store_true", help="Print dataset statistics")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.inspect_data:
        inspect_dataset_main()
    else:
        project_root = Path(__file__).resolve().parent
        print(f"Project is ready in: {project_root}")
        print("Use scripts/inspect_dataset.py or streamlit run ui/app.py")


if __name__ == "__main__":
    main()
