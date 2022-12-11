# https://github.com/simonepri/lm-scorer/blob/master/lm_scorer/models/abc/batch.py

import torch.nn as nn
import torch

# class Scorer:
#     def __init__(self, model: nn.Module, text: List[str]) -> None:
#         self.model = model

from transformers import AutoTokenizer, GPT2LMHeadModel

from transformers.tokenization_utils import BatchEncoding
from transformers import GPT2Tokenizer, GPT2LMHeadModel, T5ForConditionalGeneration, T5EncoderModel
from typing import List, Tuple
import math


# get log likelihood of all tokens in sentence for a model
def get_log_likelihoods(model, tokenizer, sentence):
    # tokenize sentence
    tokenized_text = tokenizer.tokenize(sentence)
    # convert tokens to ids
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # convert to tensor
    tokens_tensor = torch.tensor([indexed_tokens])
    # get log likelihoods
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]
    # get log likelihoods for each token
    log_likelihoods = []
    for i, token in enumerate(tokenized_text):
        # get log likelihood of token
        token_log_likelihood = predictions[0, i, indexed_tokens[i]]
        # add to list
        log_likelihoods.append(token_log_likelihood)
    return log_likelihoods


class Scorer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|pad|>"]})
        self.tokenizer.pad_token = "<|pad|>"
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()

    def _add_special_tokens(self, text: str) -> str:
        # tokenizer can have eos_token or bos_token set as None or not set at all
        if bos_token := getattr(self.tokenizer, "bos_token", None):
            text = bos_token + text
        if eos_token := getattr(self.tokenizer, "eos_token", None):
            text = text + eos_token
        return text

    def sentence_score(self, text: str) -> float:
        text_ = [text]
        text_ = list(map(self._add_special_tokens, text_))

        outputs: List[Tuple[torch.DoubleTensor, torch.LongTensor, List[str]]] = []
        if len(text_) == 0:
            return outputs

        # TODO: Handle overflowing elements for long sentences
        encoding: BatchEncoding = self.tokenizer.batch_encode_plus(
            text_,
            return_tensors="pt",
        )
        with torch.no_grad():
            ids = encoding["input_ids"].to(self.model.device)
            attention_mask = encoding["attention_mask"].to(self.model.device)
            nopad_mask = ids != self.tokenizer.pad_token_id
            logits: torch.Tensor = self.model(ids, attention_mask=attention_mask)[0]

        for sent_index in range(len(text_)):
            sent_nopad_mask = nopad_mask[sent_index]

            sent_tokens = [tok for i, tok in enumerate(encoding.tokens(sent_index)) if sent_nopad_mask[i] and i != 0]

            sent_ids = ids[sent_index, sent_nopad_mask][1:]
            sent_logits = logits[sent_index, sent_nopad_mask][:-1, :]
            sent_logits[:, self.tokenizer.pad_token_id] = float("-inf")

            # ids_scores.shape = [seq_len + 1]
            sent_ids_scores = sent_logits.gather(1, sent_ids.unsqueeze(1)).squeeze(1)
            # log_prob.shape = [seq_len + 1]
            sent_log_probs = sent_ids_scores - sent_logits.logsumexp(1)

            # breakpoint()
            # sent_log_probs = torch.DoubleTensor(sent_log_probs)
            # sent_ids = torch.LongTensor(sent_ids)

            output = (sent_log_probs, sent_ids, sent_tokens)
            outputs.append(output)

        # return outputs
        scores: List[float] = []
        reduce = "mean"
        log = False
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

        return scores[0] if isinstance(text_, str) else scores


# model_name = "gpt2"
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, add_special_tokens=False)
# model = GPT2LMHeadModel.from_pretrained(model_name)

model_name = "google/flan-t5-small"
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, add_special_tokens=False)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained(model_name)
model = T5EncoderModel.from_pretrained(model_name)
sentence = "I like this package."

scorer = Scorer(model, tokenizer)

# scores = scorer.sentence_score(sentence)
# scores1 = scorer.sentence_score("I like all packages.")


# sent_long = (
#     'Here are the most relevant elements on the webpage (links, buttons, selects and inputs) to achieve the objective below:\nObjective: buy me toothpaste from amazon\nURL: https://www.google.com/search?q=toothpaste+amazon&source=hp&ei=CpBZY5PrNsKIptQP77Se0Ag&iflsig=AJiK0e\nRelevant elements:\nlink 255 role="text" role="text" "toothpaste - Amazon.com https://www.amazon.com › toothpaste › k=toothpaste"\nlink 192 role="text" role="text" "Best Sellers in Toothpaste - Amazon.ca https://www.amazon.ca › zgbs › beauty"\nlink 148 role="heading" role="text" "Shop Amazon toothpaste - Amazon.ca Official Site Ad · https://www.amazon.ca/"\n---\nHere are the most relevant elements on the webpage (links, buttons, selects and inputs) to achieve the objective below:\nObjective: book me in for 2 at bar isabel in toronto on friday night\nURL: https://www.opentable.ca/r/bar-isabel-toronto\nRelevant elements:\nselect 119 TxpENin57omlyGS8c0YB Time selector restProfileSideBartimePickerDtpPicker "5:00 p.m. 5:30 p.m. 6:00 p.m. 6:30 p.m. 7:00 p.m. 7:30 p.m. 8:00 p.m. 8:30 p.m. 9:00 p.m. 9:30 p.m. 10:00 p.m. 10:30 p.m. 11:00 p.m. 11:30 p.m."\nselect 114 Party size selector FfVyD58WJTQB9nBaLQRB restProfileSideBarDtpPartySizePicker "1 person 2 people 3 people 4 people 5 people 6 people 7 people 8 people 9 people 10 people 11 people 12 people 13 people 14 people 15 people 16 people 17 people 18 people 19 people 20 people"\nbutton 121 aria-label="Find a time" "Find a time"\n---\nHere are the most relevant elements on the webpage (links, buttons, selects and inputs) to achieve the objective below:\nObjective: email aidan@cohere.com telling him I\'m running a few mins late\nURL: https://www.google.com/?gws_rd=ssl\nRelevant elements:\nlink 3 "Gmail"\ninput 10 gLFyf gsfi q text combobox Search Search\n---\nHere are the most relevant elements on the webpage (links, buttons, selects and inputs) to achieve the objective below:\nObjective: buy me a pair of sunglasses from amazon\nURL: https://www.amazon.ca/LUENX-Aviator-Sunglasses-Polarized-Gradient/dp/B08P7HMKJW\nRelevant elements:\nbutton 153 add-to-cart-button submit.add-to-cart Add to Shopping Cart a-button-input Add to Cart\nbutton 155 buy-now-button submit.buy-now a-button-input\nselect 152 quantity quantity a-native-dropdown a-declarative "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30"\n---\nHere are the most relevant elements on the webpage (links, buttons, selects and inputs) to achieve the objective below:\nObjective: buy bodywash\nURL: https://www.google.com/?gws_rd=ssl\nRelevant elements:\nbutton 0 L3eUgb'
#     * 1
# )
# # breakpoint()
# scores1 = scorer.sentence_score(sent_long)
# # print(scores)
# print(scores1)
# breakpoint()

# tokenizer.add_special_tokens({"additional_special_tokens": ["<|pad|>"]})
# tokenizer.pad_token = "<|pad|>"
# model.resize_token_embeddings(len(tokenizer))
# model.eval()


# log_probs = get_log_likelihoods(model, tokenizer, sentence)
# breakpoint()
# log_probs = torch.stack(log_probs)
# tlen = log_probs.shape[0]
# score = log_probs.logsumexp(0) - math.log(tlen)
# breakpoint()
