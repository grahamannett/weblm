from dataclasses import dataclass, field
from typing import List

from playwright import Page

# third party imports but ideally would have the LM extract relevant text
import bs4
from readability import Document


@dataclass
class DomTree:
    documents: List[dict] = field(default_factory=list)
    strings: List[str] = field(default_factory=list)


class LMTasks:
    prompt: str = None

    def __call__(self, text: str) -> str:
        return f"{self.prompt} {text}"

@dataclass
class Summary(LMTasks):
    prompt: str = "summarize the following pasaage:"

@dataclass
class QuestionAnswering(LMTasks):
    prompt: str = "answer the following question:"



def extract_text_using_library(content: str):
    # https://stackoverflow.com/questions/1936466/how-to-scrape-only-visible-webpage-text-with-beautifulsoup

    def tag_visible(element: bs4.element):
        if (element.parent.name in ["style", "script", "head", "title", "meta", "[document]"]) or (isinstance(element, bs4.element.Comment)):
            return False
        return True

    document = Document(content)
    title = document.title()
    summary = document.summary()
    texts = bs4.BeautifulSoup(summary, "html.parser").findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return " ".join(t.strip() for t in visible_texts)




class Parser:
    def crawl(self, tree):
        pass

    def third_party(self, page: Page):
        """"""
        pass

    def get_text_using_libraries(self, page: Page):
        content = page.content()
        text = extract_text_using_library(content)
        return text

    def summary(self, text: str):
        prompt = f"summarize the following passage:{text}"




if __name__ == "__main__":
