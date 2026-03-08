"""Capture a screenshot of the alpha phase portrait visualization."""
import asyncio
from playwright.async_api import async_playwright


async def screenshot():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1200, "height": 800})
        await page.goto(
            "file:///Users/jasonwalsh/ghq/github.com/aygp-dr/qwen3-steering"
            "/viz/interactive/alpha_phase_portrait.html"
        )
        await page.wait_for_timeout(3000)
        await page.screenshot(
            path="/Users/jasonwalsh/ghq/github.com/aygp-dr/qwen3-steering"
            "/viz/interactive/alpha_phase_portrait.png"
        )
        print("Screenshot saved to viz/interactive/alpha_phase_portrait.png")
        await browser.close()


asyncio.run(screenshot())
