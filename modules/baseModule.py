from abc import ABC, abstractmethod
import yaml
from pathlib import Path

class BaseModule(ABC):
    """
    BaseModule is the universal contract for all modules in AIPlayground.
    Every module (retriever, generator, ingestor, cache, metric, pipeline, etc.)
    must inherit from BaseModule and implement the `run` method.
    """
    _config = None  # shared across all subclasses
    
    def __init__(self, name: str = None):
        """
        Initialize a module with an optional name.
        
        Args:
            name (str, optional): Human-readable identifier for the module.
        """
        if BaseModule._config is None:  # load only once
            config_path = Path(__file__).parent.parent / "config.yaml"
            with open(config_path, "r") as f:
                BaseModule._config = yaml.safe_load(f)

        self.config = BaseModule._config
        self.name = name or self.__class__.__name__

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Abstract run method that all modules must implement.

        Args:
            *args: Positional arguments specific to the module.
            **kwargs: Keyword arguments specific to the module.

        Returns:
            Any: The output of the module.
        """
        raise NotImplementedError("Subclasses must implement the `run` method.")