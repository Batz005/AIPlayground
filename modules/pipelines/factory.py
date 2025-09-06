from modules.pipelines.qa import QAPipeline

class PipelineFactory:
    """
    Factory for creating pipeline instances from config dicts.
    Each pipeline is defined in config.yaml under `pipelines:`.
    """

    _registry = {
        "qa": QAPipeline,
        # Future pipelines: "rag": RAGPipeline, "agent": AgentPipeline, ...
    }

    @staticmethod
    def create(config: dict, pipeline_name: str):
        """
        Create a pipeline instance based on config and pipeline name.

        Args:
            config (dict): Full loaded config.yaml
            pipeline_name (str): The name of the pipeline to create (e.g., "qa")

        Returns:
            BasePipeline: An instance of the chosen pipeline.
        """
        pipelines_cfg = config.get("pipelines", {})
        if pipeline_name not in pipelines_cfg:
            raise ValueError(f"Pipeline '{pipeline_name}' not found in config")

        pipeline_cfg = pipelines_cfg[pipeline_name]
        pipeline_type = pipeline_name   # here type == name, e.g., "qa"

        if pipeline_type not in PipelineFactory._registry:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")

        cls = PipelineFactory._registry[pipeline_type]
        return cls(config, pipeline_cfg)  # pass global + specific config