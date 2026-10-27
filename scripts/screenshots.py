"""Screenshots of the console for the README (docs/screenshots), against `ptp serve` on :8000.

python scripts/screenshots.py http://127.0.0.1:8000
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"


def main(base: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=2)
        page.goto(f"{base}/")
        page.get_by_role("heading", name="Versions on the path").wait_for()
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT / "path.png"), full_page=False)
        target = sys.argv[2] if len(sys.argv) > 2 else "r0002"
        page.goto(f"{base}/models/{target}")
        page.get_by_role("heading", name="Gates").wait_for()
        page.wait_for_timeout(1200)
        page.screenshot(path=str(OUT / "model.png"), full_page=False)
        page.goto(f"{base}/evidence")
        page.wait_for_timeout(1200)
        page.screenshot(path=str(OUT / "evidence.png"), full_page=False)
        phone = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=3)
        phone.goto(f"{base}/")
        phone.wait_for_timeout(1800)
        phone.screenshot(path=str(OUT / "phone.png"), full_page=False)
        browser.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000")
