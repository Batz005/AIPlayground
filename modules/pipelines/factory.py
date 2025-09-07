from modules.pipelines.generic import GenericPipeline

class PipelineFactory:

    @staticmethod
    def create(config: dict, pipeline_name: str):
        pipelines_cfg = config.get("pipelines", {})
        if pipeline_name not in pipelines_cfg:
            raise ValueError(f"Pipeline '{pipeline_name}' not found in config")
        
        pipeline_cfg = pipelines_cfg[pipeline_name]
        return GenericPipeline(config, pipeline_cfg)