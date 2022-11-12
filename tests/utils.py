from weblm.main import WebLMState


class ScreenshotGrabber:
    def __init__(self, screenshot_dir: str = "tests/screenshots"):
        self.curr_idx = 0
        self.screenshot_dir = screenshot_dir

    def __call__(self, state: WebLMState):
        screenshot = state.crawler.page.screenshot(path=f"{self.screenshot_dir}/{self.curr_idx}.png")
        self.curr_idx += 1
