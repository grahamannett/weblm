import time
from .crawler import Crawler, replace_special_fields


class AsyncCrawler(Crawler):
    def __init__(self, playwright) -> None:
        self.playwright = playwright

    async def _init_browser(self):
        self.browser = await self.playwright.chromium.launch(
            headless=True,
        )
        self.context = await self.browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36"
        )

        self.page = await self.context.new_page()
        await self.page.set_viewport_size({"width": 1280, "height": 1080})

    async def screenshot(self):
        _path = "screenshot.png"
        await self.page.screenshot(path=_path)
        return _path

    async def crawl(self):
        start = time.time()

        page = self.page
        tree = await self.client.send(
            "DOMSnapshot.captureSnapshot",
            {"computedStyles": ["display"], "includeDOMRects": True, "includePaintOrder": True},
        )
        device_pixel_ratio = await page.evaluate("window.devicePixelRatio")
        win_scroll_x = await page.evaluate("window.scrollX")
        win_scroll_y = await page.evaluate("window.scrollY")
        win_upper_bound = await page.evaluate("window.pageYOffset")
        win_left_bound = await page.evaluate("window.pageXOffset")
        win_width = await page.evaluate("window.screen.width")
        win_height = await page.evaluate("window.screen.height")
        elements_of_interest = self._crawl(tree, win_upper_bound, win_width, win_left_bound, win_height, device_pixel_ratio)

        print("Parsing time: {:0.2f} seconds".format(time.time() - start))
        return elements_of_interest

    async def go_to_page(self, url):
        await self.page.goto(url=url if "://" in url else "http://" + url)
        self.client = await self.page.context.new_cdp_session(self.page)
        self.page_element_buffer = {}

    async def scroll(self, direction):
        if direction == "up":
            await self.page.evaluate(
                "(document.scrollingElement || document.body).scrollTop = (document.scrollingElement || document.body).scrollTop - window.innerHeight;"
            )
        elif direction == "down":
            await self.page.evaluate(
                "(document.scrollingElement || document.body).scrollTop = (document.scrollingElement || document.body).scrollTop + window.innerHeight;"
            )

    async def click(self, id):
        # Inject javascript into the page which removes the target= attribute from all links
        js = """
        links = document.getElementsByTagName("a");
        for (var i = 0; i < links.length; i++) {
            links[i].removeAttribute("target");
        }
        """
        await self.page.evaluate(js)

        element = self.page_element_buffer.get(int(id))
        if element:
            x = element.get("center_x")
            y = element.get("center_y")

            height, width = WINDOW_SIZE["height"], WINDOW_SIZE["width"]

            x_d = max(0, x - width)
            x_d += 5 * int(x_d > 0)
            y_d = max(0, y - height)
            y_d += 5 * int(y_d > 0)

            if x_d or y_d:
                await self.page.evaluate(f"() => window.scrollTo({x_d}, {y_d})")

            await self.page.mouse.click(x - x_d, y - y_d)
        else:
            print("Could not find element")

    async def type(self, id, text):
        await self.click(id)
        await self.page.evaluate(f"() => document.activeElement.value = ''")
        await self.page.keyboard.type(text)

    async def enter(self):
        await self.page.keyboard.press("Enter")

    async def run_cmd(self, cmd):
        print("cmd", cmd)
        cmd = replace_special_fields(cmd.strip())

        if cmd.startswith("SCROLL UP"):
            await self.scroll("up")
        elif cmd.startswith("SCROLL DOWN"):
            await self.scroll("down")
        elif cmd.startswith("click"):
            commasplit = cmd.split(",")
            id = commasplit[0].split(" ")[2]
            await self.click(id)
        elif cmd.startswith("type"):
            spacesplit = cmd.split(" ")
            id = spacesplit[2]
            text = spacesplit[3:]
            text = " ".join(text)
            # Strip leading and trailing double quotes
            text = text[1:-1]
            text += "\n"
            await self.type(id, text)
        else:
            raise Exception(f"Invalid command: {cmd}")

        time.sleep(2)
