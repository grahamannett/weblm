from typing import List, Any
from dataclasses import dataclass, field

from playwright.sync_api import Page
import uuid


@dataclass
class Element:
    node_index: str
    backend_node_id: int
    node_name: str
    node_value: Any
    is_clickable: bool

    origin_x: int
    origin_y: int
    center_x: int
    center_y: int

    node_meta: List[str] = field(default_factory=list)


class ElementCapture:
    def __init__(self):
        self.folder = "screenshots"

    def capture(self, page: Page, element: Element):
        screenshot = page.screenshot(
            clip={
                "x": element.origin_x,
                "y": element.origin_y, "width": element.center_x, "height": element.origin_y, },
            path=self.latest_file(),
        )
        return screenshot

    def latest_file(self):
        return f"{self.folder}/{str(uuid.uuid4())}.png"
