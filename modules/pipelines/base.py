from abc import ABC, abstractmethod
from modules.baseModule import BaseModule

class BasePipeline(BaseModule, ABC):
    """
    Abstract base class for all pipelines.
    Pipelines orchestrate multiple modules (ingestor, retriever, generator, etc.).
    """

    def __init__(self, name: str = None):
        super().__init__(name)

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Execute the pipeline flow.
        Each derived pipeline defines its own flow.
        """
        raise NotImplementedError