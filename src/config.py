from pathlib import Path
import yaml

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

def load_config(name: str) -> dict:
    path = _CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)
