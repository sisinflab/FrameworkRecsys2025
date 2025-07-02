# config.py

import yaml

def load_config(path: str = "config.yaml") -> dict:
    """
    Carica e restituisce il contenuto di config.yaml come dict.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)
