from pathlib import Path
import subprocess
import sys


def test_small_training_entrypoint(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "train_model.py"),
            "--steps", "3",
            "--batch-size", "2",
            "--context-length", "16",
            "--layers", "1",
            "--heads", "2",
            "--embedding-size", "16",
            "--dropout", "0",
            "--checkpoint-interval", "2",
            "--checkpoint", str(tmp_path / "latest.pt"),
            "--metrics", str(tmp_path / "metrics.json"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert '"vocabulary_size": 65' in result.stdout
    assert (tmp_path / "latest.pt").exists()
    assert (tmp_path / "metrics.json").exists()
