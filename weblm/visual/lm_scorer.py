from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import
from abc import ABC, abstractmethod

import math
import os

import torch
import itertools

from transformers import AutoTokenizer, GPT2LMHeadModel
from transformers import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP
from transformers.tokenization_utils import BatchEncoding


class LMScorer(ABC):
    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self._build(model_name, kwargs)

    @overload
    def sentence_score(self, text: str, log: bool = False, reduce: str = "prod") -> float:
        ...

    @overload
    def sentence_score(self, text: List[str], log: bool = False, reduce: str = "prod") -> List[float]:
        ...

    def sentence_score(
        self,
        text: Union[str, List[str]],
        log: bool = False,
        reduce: str = "prod",
    ) -> Union[float, List[float]]:
        sentences = [text] if isinstance(text, str) else text
        scores: List[float] = []
        if len(sentences) == 0:
            return scores

        outputs = self._tokens_log_prob(sentences)
        for output in outputs:
            log_probs = output[0]
            tlen = log_probs.shape[0]

            if reduce == "prod":
                score = log_probs.sum()
            elif reduce == "mean":
                score = log_probs.logsumexp(0) - math.log(tlen)
            elif reduce == "gmean":
                score = log_probs.mean(0)
            elif reduce == "hmean":
                score = log_probs.neg().logsumexp(0).neg() + math.log(tlen)
            else:
                raise ValueError("Unrecognized scoring strategy: %s" % reduce)
            if not log:
                score = score.exp()

            scores.append(score.item())

        return scores[0] if isinstance(text, str) else scores

    @overload
    def tokens_score(self, text: str, log: bool = False) -> Tuple[List[float], List[int], List[str]]:
        ...

    @overload
    def tokens_score(self, text: List[str], log: bool = False) -> List[Tuple[List[float], List[int], List[str]]]:
        ...

    def tokens_score(
        self, text: Union[str, List[str]], log: bool = False
    ) -> Union[Tuple[List[float], List[int], List[str]], List[Tuple[List[float], List[int], List[str]]],]:
        sentences = [text] if isinstance(text, str) else text
        outputs: List[Tuple[List[float], List[int], List[str]]] = []
        if len(sentences) == 0:
            return outputs

        for log_probs, ids, tokens in self._tokens_log_prob(sentences):
            scores = log_probs if log else log_probs.exp()
            scores = cast(torch.DoubleTensor, scores)
            output = (scores.tolist(), ids.tolist(), tokens)
            outputs.append(output)

        return outputs[0] if isinstance(text, str) else outputs

    @classmethod
    def supported_model_names(cls) -> Iterable[str]:
        return cls._supported_model_names()

    def _build(self, model_name: str, options: Dict[str, Any]) -> None:
        # pylint: disable=attribute-defined-outside-init, unused-argument
        self.model_name = model_name

    @abstractmethod
    def _tokens_log_prob(self, text: List[str]) -> List[Tuple[torch.DoubleTensor, torch.LongTensor, List[str]]]:
        ...  # pragma: no cover

    @classmethod
    @abstractmethod
    def _supported_model_names(cls) -> Iterable[str]:
        ...  # pragma: no cover


class BatchedLMScorer(LMScorer):
    # @overrides
    def _build(self, model_name: str, options: Dict[str, Any]) -> None:
        super()._build(model_name, options)

        batch_size = options.get("batch_size", 1)
        if batch_size < 1:
            raise ValueError("The batch_size option must be positive")
        # pylint: disable=attribute-defined-outside-init
        self.batch_size = batch_size

    # @overrides
    def _tokens_log_prob(self, text: List[str]) -> List[Tuple[torch.DoubleTensor, torch.LongTensor, List[str]]]:
        outputs = []
        for i in range(0, len(text), self.batch_size):
            batch = text[i : i + self.batch_size]
            outputs.extend(self._tokens_log_prob_for_batch(batch))
        return outputs

    @abstractmethod
    def _tokens_log_prob_for_batch(self, text: List[str]) -> List[Tuple[torch.DoubleTensor, torch.LongTensor, List[str]]]:
        ...  # pragma: no cover


class TransformersLMScorer(BatchedLMScorer):
    # @overrides
    def _build(self, model_name: str, options: Dict[str, Any]) -> None:
        super()._build(model_name, options)

        #  Make transformers cache path configurable.
        cache_dir = os.environ.get("TRANSFORMERS_CACHE_DIR", ".transformers_cache")
        options["cache_dir"] = options.get("cache_dir", cache_dir)


class GPT2LMScorer(TransformersLMScorer):
    # @overrides
    def _build(self, model_name: str, options: Dict[str, Any]) -> None:
        super()._build(model_name, options)

        # pylint: disable=attribute-defined-outside-init
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, add_special_tokens=False)
        # Add the pad token to GPT2 dictionary.
        # len(tokenizer) = vocab_size + 1
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|pad|>"]})
        self.tokenizer.pad_token = "<|pad|>"

        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        # We need to resize the embedding layer because we added the pad token.
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()
        if "device" in options:
            self.model.to(options["device"])

    def _add_special_tokens(self, text: str) -> str:
        return self.tokenizer.bos_token + text + self.tokenizer.eos_token

    # @overrides
    def _tokens_log_prob_for_batch(self, text: List[str]) -> List[Tuple[torch.DoubleTensor, torch.LongTensor, List[str]]]:
        outputs: List[Tuple[torch.DoubleTensor, torch.LongTensor, List[str]]] = []
        if len(text) == 0:
            return outputs

        # TODO: Handle overflowing elements for long sentences
        # text = list(map(self._add_special_tokens, text))
        encoding: BatchEncoding = self.tokenizer.batch_encode_plus(
            text,
            return_tensors="pt",
        )
        with torch.no_grad():
            ids = encoding["input_ids"].to(self.model.device)
            attention_mask = encoding["attention_mask"].to(self.model.device)
            nopad_mask = ids != self.tokenizer.pad_token_id
            logits: torch.Tensor = self.model(ids, attention_mask=attention_mask)[0]

        for sent_index in range(len(text)):
            sent_nopad_mask = nopad_mask[sent_index]
            # len(tokens) = len(text[sent_index]) + 1
            sent_tokens = [tok for i, tok in enumerate(encoding.tokens(sent_index)) if sent_nopad_mask[i] and i != 0]

            # sent_ids.shape = [len(text[sent_index]) + 1]
            sent_ids = ids[sent_index, sent_nopad_mask][1:]
            # logits.shape = [len(text[sent_index]) + 1, vocab_size]
            sent_logits = logits[sent_index, sent_nopad_mask][:-1, :]
            sent_logits[:, self.tokenizer.pad_token_id] = float("-inf")

            # ids_scores.shape = [seq_len + 1]
            sent_ids_scores = sent_logits.gather(1, sent_ids.unsqueeze(1)).squeeze(1)
            # log_prob.shape = [seq_len + 1]
            sent_log_probs = sent_ids_scores - sent_logits.logsumexp(1)

            sent_log_probs = cast(torch.DoubleTensor, sent_log_probs)
            sent_ids = cast(torch.LongTensor, sent_ids)

            output = (sent_log_probs, sent_ids, sent_tokens)
            outputs.append(output)

        return outputs

    # @overrides
    @classmethod
    def _supported_model_names(cls) -> Iterable[str]:
        return GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()


class AutoLMScorer:
    MODEL_CLASSES = [GPT2LMScorer]

    def __init__(self):
        raise EnvironmentError(
            "AutoLMscorer is designed to be instantiated " "using the `AutoLMscorer.from_pretrained(model_name)`" "method"
        )

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs: Any) -> LMScorer:
        for model_class in cls.MODEL_CLASSES:
            if model_name not in model_class.supported_model_names():
                continue
            return model_class(model_name, **kwargs)
        raise ValueError(
            "Unrecognized model name." "Can be one of: %s" % ", ".join(cls.supported_model_names()),
        )

    @classmethod
    def supported_model_names(cls) -> Iterable[str]:
        classes = cls.MODEL_CLASSES
        models = map(lambda c: c.supported_model_names(), classes)
        return itertools.chain.from_iterable(models)


# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# batch_size = 1
# scorer = AutoLMScorer.from_pretrained("gpt2", device=device, batch_size=batch_size)

# sentence = "I like this package."
# out = scorer.sentence_score(sentence, reduce="mean")
# print(out)


# log_probs
# tensor([-3.9997, -5.0142, -2.5179, -7.4062, -1.2812, -5.6162])
