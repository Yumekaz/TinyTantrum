from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    seed: int = 1337
    data_dir: Path = Path("data")
    dataset_name: str = "tinyshakespeare.txt"
    dataset_url: str = (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    )
    expected_sha256: str | None = None

    @property
    def dataset_path(self) -> Path:
        return self.data_dir / self.dataset_name
