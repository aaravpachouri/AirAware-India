"""Download and verify the original India city-day AQI dataset.

The raw CSV is intentionally ignored by Git. The checksum pins the expected
source file so CI does not silently evaluate a changed or unrelated dataset.
"""
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "city_day.csv"
SOURCE_URL = (
    "https://raw.githubusercontent.com/learning-monk/datasets/master/"
    "ENVIRONMENT/Indian_cities_daily_pollution_2015-2020.csv"
)
EXPECTED_SHA256 = "0d84b21c3e4878bbad8df362f2ab05f61ad959538dddf5918e714077ed3c1847"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading source dataset from {SOURCE_URL}")
    with urlopen(SOURCE_URL, timeout=60) as response:
        payload = response.read()

    actual_hash = sha256(payload).hexdigest()
    if actual_hash != EXPECTED_SHA256:
        raise ValueError(
            "Source dataset checksum mismatch. "
            f"Expected {EXPECTED_SHA256}, received {actual_hash}. "
            "Do not train on an unverified dataset."
        )

    DATA_PATH.write_bytes(payload)
    print(f"Verified source dataset SHA-256: {actual_hash}")
    print(f"Saved {len(payload):,} bytes to {DATA_PATH}")


if __name__ == "__main__":
    main()
