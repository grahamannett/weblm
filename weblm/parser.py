from dataclasses import dataclass
from typing import List

from .tasks_stubs import summary_stub

import bs4
from readability import Document


class LMTasks:
    prompt: str = None

    def __call__(self, text: str, **kwargs) -> str:
        return self.prompt.format(text=text, **kwargs)


@dataclass
class SummaryTask(LMTasks):
    prompt = summary_stub


@dataclass
class QuestionAnsweringTask(LMTasks):
    prompt: str = "passage:{text}\nanswer the following question:{question}"


class TasksInterface:
    summary = SummaryTask()
    qa = QuestionAnsweringTask()


def extract_text_using_library(content: str) -> str:
    # https://stackoverflow.com/questions/1936466/how-to-scrape-only-visible-webpage-text-with-beautifulsoup

    _visible_tags = ["style", "script", "head", "title", "meta", "[document]"]

    document = Document(content)
    title = document.title()
    summary = document.summary()
    texts = bs4.BeautifulSoup(summary, "html.parser").findAll(text=True)

    visible_texts = [t for t in texts if t.parent.name not in _visible_tags and not isinstance(t, bs4.element.Comment)]
    return " ".join(t.strip() for t in visible_texts)


class Parser:
    def __init__(self, content: str):
        self.content = content

    def process(self) -> str:
        return extract_text_using_library(self.content)

    def extract_text_using_lm(self, content: str):
        raise NotImplementedError("extract_text_using_lm not implemented yet")
