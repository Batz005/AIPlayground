from transformers import pipeline
from modules.generators.base import BaseGenerator

class FlanT5Generator(BaseGenerator):
    def __init__(self, model_name: str = "google/flan-t5-base",
                 max_new_tokens: int = 64, temperature: float = 0.0, device: str | None = None):
        super().__init__(name="FlanT5Generator", max_new_tokens=max_new_tokens, temperature=temperature)
        # use device from config if not passed (e.g., "mps" or "cpu")
        device = device or self.config["embedding"].get("device", "cpu")
        # HF pipeline handles device mapping automatically; keep it simple
        self.pipe = pipeline("text2text-generation", model=model_name, device_map="auto")

    def _build_prompt(self, query: str, contexts: list[str]) -> str:
        # Use just top 3 contexts to keep prompt tight
        ctx = "\n\n---\n\n".join(contexts[:2])

        return (
            "You are a helpful assistant. "
            "Using the context below, select the ones that seem relevant and based on that, answer the question in clear, natural and fluent English, using a short paragraph."
            "If the exact answer is not found, provide the closest relevant information from the context. "
            "If nothing is even remotely relevant, say 'I don't know.'\n\n"
            f"Context:\n{ctx}\n\n"
            f"Question: {query}\n"
            "Answer in as few sentences as possible. Do not repeat yourself.:"
        )
        

    def generate(self, query: str, contexts: list[str]) -> str:
        prompt = self._build_prompt(query, contexts)
        out = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,          # sampling instead of greedy
            top_p=0.9,               # nucleus sampling
            repetition_penalty=1.2   # reduce looping
        )
        return out[0]["generated_text"].strip()