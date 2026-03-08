import py_compile
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    files = [
        root / "train.py",
        root / "engine.py",
        root / "metrics.py",
        root / "utils.py",
        root / "datasets" / "__init__.py",
        root / "datasets" / "segmentation_dataset.py",
        root / "losses" / "__init__.py",
        root / "losses" / "segmentation_losses.py",
    ]
    for file in files:
        py_compile.compile(str(file), doraise=True)
    print("Training stack compile test passed.")


if __name__ == "__main__":
    main()
