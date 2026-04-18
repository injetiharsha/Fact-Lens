"""Prepare datasets for training tasks."""

from scripts.prepare_context_data import main as prepare_context_main


def prepare_datasets():
    """Build training-ready processed datasets."""
    print("Preparing context datasets...")
    prepare_context_main()


if __name__ == "__main__":
    prepare_datasets()
