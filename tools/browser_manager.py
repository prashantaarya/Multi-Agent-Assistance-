"""
Playwright Browser Manager — production-grade browser lifecycle.

Features:
- Singleton browser instance (one Chromium process for all scrapers)
- Per-task BrowserContext (isolated cookies/storage per scrape)
- Stealth mode (anti bot-detection)
- Persistent storage state per site (login sessions cached)
- Graceful shutdown
- Async-safe with locks
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

logger = logging.getLogger("jarvis.browser")

# ── Stealth user agent (rotated occasionally) ───────────────────────────────
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

# ── Default viewport (desktop) ──────────────────────────────────────────────
DEFAULT_VIEWPORT = {"width": 1920, "height": 1080}

# ── Storage state directory (persistent logins) ─────────────────────────────
STORAGE_DIR = Path("data/browser_storage")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


class BrowserManager:
    """
    Singleton browser manager.

    Usage:
        bm = BrowserManager.instance()
        async with bm.new_context(site="linkedin") as ctx:
            page = await ctx.new_page()
            await page.goto("https://linkedin.com/jobs")
    """

    _instance: Optional["BrowserManager"] = None
    _lock = asyncio.Lock()

    def __init__(self) -> None:
        self._pw: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._init_lock = asyncio.Lock()
        self._headless = os.getenv("PLAYWRIGHT_HEADLESS", "true").lower() != "false"
        self._slow_mo_ms = int(os.getenv("PLAYWRIGHT_SLOW_MO_MS", "0"))

    # ── Singleton ──────────────────────────────────────────────────────────
    @classmethod
    def instance(cls) -> "BrowserManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Lifecycle ──────────────────────────────────────────────────────────
    async def _ensure_started(self) -> None:
        """Lazily start Playwright + browser (thread-safe)."""
        if self._browser is not None:
            return
        async with self._init_lock:
            if self._browser is not None:
                return

            self._pw = await async_playwright().start()

            # Use system Chrome if PLAYWRIGHT_USE_SYSTEM_CHROME=true (corporate networks
            # often block the Chromium download). Otherwise use bundled Chromium.
            use_system = os.getenv("PLAYWRIGHT_USE_SYSTEM_CHROME", "true").lower() == "true"
            launch_kwargs = {
                "headless": self._headless,
                "slow_mo": self._slow_mo_ms,
                "args": [
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ],
            }

            if use_system:
                # Try channels in order: chrome, msedge, then fallback to bundled
                for channel in ("chrome", "msedge"):
                    try:
                        logger.info("Launching system %s (headless=%s)", channel, self._headless)
                        self._browser = await self._pw.chromium.launch(
                            channel=channel, **launch_kwargs
                        )
                        return
                    except Exception as e:
                        logger.warning("System %s unavailable: %s", channel, str(e)[:200])

            logger.info("Launching bundled Chromium (headless=%s)", self._headless)
            self._browser = await self._pw.chromium.launch(**launch_kwargs)

    async def shutdown(self) -> None:
        """Cleanly close browser + playwright."""
        if self._browser:
            try:
                await self._browser.close()
            except Exception as e:
                logger.warning("Error closing browser: %s", e)
            self._browser = None
        if self._pw:
            try:
                await self._pw.stop()
            except Exception as e:
                logger.warning("Error stopping playwright: %s", e)
            self._pw = None

    # ── Context factory ────────────────────────────────────────────────────
    @asynccontextmanager
    async def new_context(
        self,
        site: str = "generic",
        persist_storage: bool = True,
        user_agent: str = DEFAULT_USER_AGENT,
        timezone_id: str = "Asia/Kolkata",
        locale: str = "en-US",
        extra_headers: Optional[dict] = None,
    ) -> AsyncIterator[BrowserContext]:
        """
        Create an isolated BrowserContext for a single scrape task.

        Args:
            site: Identifier for persistent storage (e.g., "linkedin", "naukri")
            persist_storage: Reuse cookies/localStorage from previous sessions
        """
        await self._ensure_started()
        assert self._browser is not None

        storage_path = STORAGE_DIR / f"{site}.json" if persist_storage else None
        storage_state = (
            str(storage_path) if storage_path and storage_path.exists() else None
        )

        context = await self._browser.new_context(
            user_agent=user_agent,
            viewport=DEFAULT_VIEWPORT,
            locale=locale,
            timezone_id=timezone_id,
            storage_state=storage_state,
            extra_http_headers=extra_headers or {},
        )

        # Inject stealth scripts before any page load
        await self._inject_stealth(context)

        try:
            yield context
        finally:
            # Persist cookies/storage on the way out
            if storage_path:
                try:
                    await context.storage_state(path=str(storage_path))
                except Exception as e:
                    logger.warning("Could not save storage state for %s: %s", site, e)
            await context.close()

    @asynccontextmanager
    async def new_page(self, site: str = "generic", **ctx_kwargs) -> AsyncIterator[Page]:
        """Shortcut: context + page in one call."""
        async with self.new_context(site=site, **ctx_kwargs) as ctx:
            page = await ctx.new_page()
            page.set_default_timeout(30_000)
            page.set_default_navigation_timeout(45_000)
            yield page

    # ── Stealth injection ──────────────────────────────────────────────────
    @staticmethod
    async def _inject_stealth(context: BrowserContext) -> None:
        """
        Inject anti-detection scripts.
        Removes webdriver flag, spoofs navigator props, hides automation.
        """
        await context.add_init_script(
            """
            // Remove webdriver flag
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

            // Spoof plugins
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5].map(() => ({ name: 'Chrome PDF Plugin' })),
            });

            // Spoof languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
            });

            // Spoof hardware concurrency
            Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 8 });

            // Remove headless markers from Chrome runtime
            window.chrome = { runtime: {}, loadTimes: () => {}, csi: () => {} };

            // Permissions API spoof
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications'
                    ? Promise.resolve({ state: Notification.permission })
                    : originalQuery(parameters)
            );
            """
        )


# ── Module-level shutdown hook (call from app shutdown) ─────────────────────
async def shutdown_browser() -> None:
    """Convenience hook for FastAPI shutdown events."""
    await BrowserManager.instance().shutdown()
