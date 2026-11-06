"""Write the synthetic SABR datasets to ``data/``."""

from pathlib import Path

from waml.data import write_datasets

ROOT = Path(__file__).resolve().parents[1]
NUM_SEEDS = 5


if __name__ == "__main__":
    write_datasets(ROOT / "data", range(NUM_SEEDS))
    print(f"Wrote datasets to {ROOT / 'data'}")
