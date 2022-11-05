from typing import List

import cohere
import numpy as np

from .base import BaseController, truncate_left


class CohereController(BaseController):
    MODEL = "xlarge"
    cohere_client = None

    def __init__(self, co: cohere.Client, objective: str, **kwargs):
        super().__init__(objective)
        self.co = co

        if CohereController.cohere_client is None:
            CohereController.cohere_client = co

        self._fn = self.make_fn()

    def embed(self, texts: List[str], truncate: str = "RIGHT") -> cohere.embeddings.Embeddings:
        return self.co.embed(texts=texts, truncate=truncate)

    def generate(
        self,
        prompt: str,
        model: str = None,
        temperature: float = 0.5,
        num_generations: int = 5,
        max_tokens: int = 20,
        stop_sequences: List[str] = ["\n"],
        return_likelihoods: str = "GENERATION",
    ) -> cohere.generation.Generations:

        return self.co.generate(
            prompt=prompt,
            model=model if model else self.MODEL,
            temperature=temperature,
            num_generations=num_generations,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            return_likelihoods="GENERATION",
        )

    def tokenize(self, text: str) -> cohere.tokenize.Tokens:
        return self.co.tokenize(text=text)

    def make_fn(self):
        def _fn(x):
            if len(x) == 3:
                option, prompt, self = x
                return_likelihoods = "ALL"
            elif len(x) == 4:
                option, prompt, self, return_likelihoods = x

            while True:
                try:
                    if len(self.co.tokenize(prompt)) > 2048:
                        prompt = truncate_left(self.tokenize, prompt)
                    return (
                        self.generate(prompt=prompt, max_tokens=0, model=self.MODEL, return_likelihoods=return_likelihoods)
                        .generations[0]
                        .likelihood,
                        option,
                    )
                except cohere.error.CohereError as e:
                    print(f"Cohere fucked up: {e}")
                    continue
                except ConnectionError as e:
                    print(f"Connection error: {e}")
                    continue

        return _fn
