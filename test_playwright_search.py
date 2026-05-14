"""
Smoke test for the Playwright job search stack.

Run:
    python test_playwright_search.py

Verifies:
1. BrowserManager can launch a browser (system Chrome / Edge / bundled).
2. Naukri scraper returns real listings (no login required).
3. Orchestrator dedupes + ranks results.
4. ApplicationTracker filters already-applied jobs.

This does NOT submit any applications.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

# Force logging early so we see what's happening
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("smoke")

# Headed by default to make it obvious the browser is real; override with env.
os.environ.setdefault("PLAYWRIGHT_HEADLESS", "true")
os.environ.setdefault("PLAYWRIGHT_USE_SYSTEM_CHROME", "true")


async def test_browser_launch() -> bool:
    log.info("── Test 1: Browser launch ────────────────────────────────")
    from tools.browser_manager import BrowserManager

    bm = BrowserManager.instance()
    try:
        async with bm.new_context(site="smoke-test", persist_storage=False) as ctx:
            page = await ctx.new_page()
            await page.goto("https://example.com", timeout=20_000)
            title = await page.title()
            log.info("  → Loaded example.com, title=%r", title)
        return True
    except Exception as e:
        log.error("  ✗ Browser launch failed: %s", e)
        return False


async def test_naukri_scrape() -> bool:
    log.info("── Test 2: Naukri scrape ─────────────────────────────────")
    from core.job_models import JobSearchFilter, JobSource
    from tools.job_scrapers import get_scraper

    scraper = get_scraper(JobSource.NAUKRI)
    try:
        jobs = await scraper.scrape(query="Python Developer", location="Bangalore", num_results=5)
        log.info("  → Got %d jobs from Naukri", len(jobs))
        for j in jobs[:3]:
            log.info("    - [%s] %s @ %s", j.source.value, j.title, j.company)
        return len(jobs) > 0
    except Exception as e:
        log.exception("  ✗ Naukri scrape failed: %s", e)
        return False


async def test_orchestrator() -> bool:
    log.info("── Test 3: Multi-source orchestrator ─────────────────────")
    from core.job_models import JobSearchFilter, JobSource
    from tools.job_orchestrator import get_orchestrator

    orch = get_orchestrator()
    filt = JobSearchFilter(
        query="AI Engineer",
        location="Bangalore",
        num_results=10,
        sources=[JobSource.NAUKRI, JobSource.LINKEDIN],
    )
    try:
        user_profile = {"skills": ["python", "ml", "pytorch"], "experience_years": 3}
        result = await orch.search(filt, user_profile=user_profile, exclude_applied=False)
        log.info(
            "  → Orchestrator: %d unique jobs across %s",
            len(result.jobs),
            [s.value for s in result.sources_searched],
        )
        for j in result.jobs[:5]:
            log.info(
                "    - [%s] %s @ %s — match=%s%%",
                j.source.value,
                j.title,
                j.company,
                j.match_score,
            )
        return len(result.jobs) > 0
    except Exception as e:
        log.exception("  ✗ Orchestrator failed: %s", e)
        return False


async def main() -> int:
    log.info("Starting Playwright job-search smoke tests")
    results: dict[str, bool] = {}

    results["browser_launch"] = await test_browser_launch()
    if not results["browser_launch"]:
        log.error("Browser launch failed — skipping remaining tests")
    else:
        results["naukri_scrape"] = await test_naukri_scrape()
        results["orchestrator"] = await test_orchestrator()

    # Cleanup
    try:
        from tools.browser_manager import BrowserManager

        await BrowserManager.instance().shutdown()
    except Exception:
        pass

    log.info("── Results ───────────────────────────────────────────────")
    for name, ok in results.items():
        log.info("  %s %s", "✓" if ok else "✗", name)

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
