import base64

# import torch.nn as nn
import numpy as np
from playwright.sync_api import Page
from transformers import MobileViTFeatureExtractor, MobileViTModel

from ..page_elements import Element, ElementCapture


class VisualCrawler:
    def __init__(self):
        self.model = MobileViTModel.from_pretrained("apple/mobilevit-small")
        self.element_capture = ElementCapture()

    def capture_page(self, page: Page):
        screenshot_buffer = page.screenshot()
        screenshot_buffer = base64.b64encode(screenshot_buffer).decode()
        return screenshot_buffer

    def find_element_in_page(self, page: Page, element: Element):
        page_buffer = self.capture_page(page)
        element = self.element_capture.capture(page, element)
        return element, page_buffer

    def match_element(self, page_buffer, element_buffer):
        pass
