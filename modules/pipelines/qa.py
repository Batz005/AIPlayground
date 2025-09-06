from modules.pipelines.base import BasePipeline
from modules.ingestors.factory import IngestorFactory
from modules.retrievers.factory import RetrieverFactory
from modules.ingestors.base import BaseIngestor
from modules.retrievers.base import BaseRetriever

# (GeneratorFactory will come once you define generators)

class QAPipeline(BasePipeline):
    """
    QA pipeline: builds modules from a sequence defined in config.
    Example sequence: ["ingestor:advanced", "retriever:semantic"]
    """

    def __init__(self, global_config: dict, pipeline_config: dict):
        super().__init__(name="QAPipeline")
        self.global_config = global_config
        self.sequence = pipeline_config.get("sequence", [])

        # build modules in order
        self.modules = []
        for step in self.sequence:
            module_type, variant = step.split(":")
            module_cfg = global_config["modules"][module_type][variant]

            if module_type == "ingestor":
                self.modules.append(IngestorFactory.create({"type": variant, **module_cfg}))
            elif module_type == "retriever":
                self.modules.append(RetrieverFactory.create({"type": variant, **module_cfg}))
            # elif module_type == "generator":
            #     self.modules.append(GeneratorFactory.create({"type": variant, **module_cfg}))
            else:
                raise ValueError(f"Unsupported module type: {module_type}")

    def run(self, query: str, folder: str):
        """
        Run the QA pipeline:
        - ingest → retrieve (→ generate in future)
        """
        ingestor = next(m for m in self.modules if isinstance(m, BaseIngestor))
        retriever = next(m for m in self.modules if isinstance(m, BaseRetriever))

        # Step 1. Ingest docs
        chunks = ingestor.run(folder)

        # Step 2. Retrieve top chunks
        results = retriever.run(query, chunks)

        return results