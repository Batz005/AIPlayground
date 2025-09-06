from modules.retrievers.semanticRetriever import SemanticRetriever
from modules.retrievers.keywordRetriever import KeywordRetriever

class RetrieverFactory:
    _registry = {
        "semantic": SemanticRetriever,
        "keyword": KeywordRetriever,
    }

    @staticmethod
    def create(config: dict):
        retriever_type = config.get("type")
        if retriever_type not in RetrieverFactory._registry:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
        cls = RetrieverFactory._registry[retriever_type]
        return cls(**{k: v for k, v in config.items() if k != "type"})