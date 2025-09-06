from typing import List
from sentence_transformers import SentenceTransformer
import yaml
from pathlib import Path
from functools import lru_cache


# Load config
with open(Path(__file__).parent.parent / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize default model once
_default_model = SentenceTransformer(config["embedding"]["model_name"], device=config["embedding"]["device"])

@lru_cache(maxsize=3)
def get_model(model_name: str):
    """Load and cache models by name."""
    return SentenceTransformer(model_name, device=config["embedding"]["device"])

def vectorize_string(txt: str, model_name: str = None):
    """Return embedding for a single string. Uses default model unless model_name is provided."""
    model = _default_model if model_name is None else get_model(model_name)
    return model.encode(txt)

def vectorize_all(texts: List[str], model_name: str = None):
    """Return embeddings for a list of strings. Uses default model unless model_name is provided."""
    model = _default_model if model_name is None else get_model(model_name)
    return model.encode(texts)