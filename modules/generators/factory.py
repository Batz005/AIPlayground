from modules.generators.flanT5 import FlanT5Generator

class GeneratorFactory:
    _registry = {
        "flan_t5_small": lambda **kwargs: FlanT5Generator(model_name="google/flan-t5-small", **kwargs),
        "flan_t5_base": lambda **kwargs: FlanT5Generator(model_name="google/flan-t5-base", **kwargs),
        "flan_alpaca_base": lambda **kwargs: FlanT5Generator(model_name="declare-lab/flan-alpaca-base", **kwargs),
    }

    @staticmethod
    def create(config: dict):
        gen_type = config.get("type")
        if gen_type not in GeneratorFactory._registry:
            raise ValueError(f"Unknown generator type: {gen_type}")
        factory = GeneratorFactory._registry[gen_type]
        return factory(**{k: v for k, v in config.items() if k != "type"})