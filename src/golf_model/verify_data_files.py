import sys
from pathlib import Path
from typing import Dict, List, Tuple

from .data_loader import PGATourDataLoader
from .config.model_config import DEFAULT_DATA_PATH


def _status(ok: bool) -> str:
    return "✅" if ok else "❌"


def verify(data_path: str = None) -> Dict:
    data_path = str(DEFAULT_DATA_PATH if data_path is None else Path(data_path))
    loader = PGATourDataLoader(data_path)

    years = [2023, 2024, 2025]
    report: Dict[int, Dict[str, str]] = {}

    print("\n" + "=" * 80)
    print("VERIFY DATA FILES")
    print(f"Data path: {data_path}")
    print("=" * 80)

    p = Path(data_path)
    if not p.exists():
        print(f"❌ Folder not found: {p}")
        return {"error": "data_path_missing"}

    xlsx = sorted(p.glob("*.xlsx"))
    print(f"Found {len(xlsx)} .xlsx files")
    print("-" * 80)

    for year in years:
        report[year] = {}
        print(f"\nYEAR {year}")
        print("-" * 80)

        ok_count = 0
        for stat_key in loader.stat_keys:
            patterns = loader.pattern_map.get(stat_key, [stat_key])
            f = loader._find_xlsx(year, patterns)  # intentional internal call for debugging
            ok = f is not None
            report[year][stat_key] = str(f.name) if f else ""
            print(f"{_status(ok)} {stat_key:16s} -> {f.name if f else 'NOT FOUND'}")
            ok_count += 1 if ok else 0

        print(f"\nSummary {year}: {ok_count}/{len(loader.stat_keys)} stats found")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80 + "\n")
    return report


if __name__ == "__main__":
    # Allow: python -m golf_model.verify_data_files [optional_path]
    path = sys.argv[1] if len(sys.argv) > 1 else None
    verify(path)
from pathlib import Path

REQUIRED_KEYWORDS = [
    "SG",
    "OTT",
    "APP",
    "ARG",
    "PUTT"
]

def verify_data_files(data_path: str) -> None:
    base = Path(data_path)
    if not base.exists():
        raise FileNotFoundError(f"Data path does not exist: {base}")

    files = list(base.rglob("*.xlsx"))
    if not files:
        raise FileNotFoundError("No .xlsx files found in data directory")

    print(f"[VERIFY] Found {len(files)} Excel files")

    hits = {k: False for k in REQUIRED_KEYWORDS}
    for f in files:
        name = f.name.upper()
        for k in hits:
            if k in name:
                hits[k] = True

    missing = [k for k, v in hits.items() if not v]
    if missing:
        print(f"[VERIFY WARNING] Missing expected stat types: {missing}")
    else:
        print("[VERIFY] Core stat files detected")
