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
    expected_sha256: str | None = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"

    @property
    def dataset_path(self) -> Path:
        return self.data_dir / self.dataset_name
