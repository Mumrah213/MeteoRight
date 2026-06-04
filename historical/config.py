
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class DownloadConfig:
    """Configuration for a weather data download.

    All fields have sensible defaults. Create with named args:
        config = DownloadConfig(lat=55.6, lon=13.0, start="2024-03-01", end="2024-03-02")
    """

    latitude: float
    longitude: float
    start_date: str
    end_date: str
    variables: tuple[str, ...] = field(default_factory=lambda: ("temperature_2m", "precipitation"))
    model: str | None = None
    output_dir: Path = field(default_factory=lambda: Path("./data"))
    max_retries: int = 3
    retry_delay: float = 1.0
    api_key: str | None = None

    @property
    def data_type(self) -> Literal["forecast", "observations"]:
        """Infer data type from model being used."""
        if self.model:
            return "forecast"
        return "observations"
