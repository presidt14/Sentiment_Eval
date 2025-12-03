import pandas as pd
from pathlib import Path

def load_posts(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    elif path.suffix.lower() in (".json", ".jsonl"):
        return pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

def load_labels(path: str | Path) -> pd.DataFrame:
    return load_posts(path)
