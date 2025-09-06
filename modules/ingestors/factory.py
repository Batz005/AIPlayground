from modules.ingestors.simpleIngestor import SimpleIngestor
from modules.ingestors.advancedIngestor import AdvancedIngestor


class IngestorFactory:
    """
    Factory for creating ingestor instances from config dicts.
    """

    _registry = {
        "simple": SimpleIngestor,
        "advanced": AdvancedIngestor,
    }

    @staticmethod
    def create(config: dict):
        """
        Create an ingestor based on config.

        Args:
            config (dict): Must include key "type" and optional params.

        Returns:
            BaseIngestor: An instance of the chosen ingestor.
        """
        ingestor_type = config.get("type")
        if ingestor_type not in IngestorFactory._registry:
            raise ValueError(f"Unknown ingestor type: {ingestor_type}")
        cls = IngestorFactory._registry[ingestor_type]
        return cls(**{k: v for k, v in config.items() if k != "type"})