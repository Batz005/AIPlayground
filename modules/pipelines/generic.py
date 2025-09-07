from modules.pipelines.base import BasePipeline
from modules.ingestors.base import BaseIngestor
from modules.retrievers.base import BaseRetriever
from modules.generators.base import BaseGenerator
from modules.ingestors.factory import IngestorFactory
from modules.retrievers.factory import RetrieverFactory
from modules.generators.factory import GeneratorFactory


# ... imports remain

class GenericPipeline(BasePipeline):
    def __init__(self, global_config: dict, pipeline_config: dict):
        super().__init__(name="GenericPipeline")
        self.global_config = global_config
        self.sequence = pipeline_config.get("sequence", [])
        self.modules = []
        self.last_chunks = None
        self.last_retrieval = None

        for step in self.sequence:
            module_type, variant = step.split(":")
            module_cfg = global_config["modules"][module_type][variant]
            if module_type == "ingestor":
                self.modules.append(IngestorFactory.create({"type": variant, **module_cfg}))
            elif module_type == "retriever":
                self.modules.append(RetrieverFactory.create({"type": variant, **module_cfg}))
            elif module_type == "generator":
                self.modules.append(GeneratorFactory.create({"type": variant, **module_cfg}))
            else:
                raise ValueError(f"Unsupported module type: {module_type}")

    def run(self, query: str, folder: str):
        data = folder
        for module in self.modules:
            if isinstance(module, BaseIngestor):
                data = module.run(data)
                self.last_chunks = data
            elif isinstance(module, BaseRetriever):
                data = module.run(query, data)
                self.last_retrieval = data
            elif isinstance(module, BaseGenerator):
                # pass only the texts from retrieval
                contexts = [chunk for chunk, _ in (self.last_retrieval or [])]
                data = module.run(query, contexts)
            else:
                raise ValueError(f"Unsupported module in sequence: {module}")
        return data