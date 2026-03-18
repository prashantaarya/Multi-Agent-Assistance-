# agents/api_manager.py
"""
API Manager with Industry-Standard Error Handling

Features:
- Retry with exponential backoff
- Circuit breaker pattern
- Structured error types
- Timeout handling
"""

import os
import asyncio
import aiohttp
import logging

from core.errors import (
    APIConnectionError,
    APITimeoutError,
    APIRateLimitError,
    APIInvalidResponseError,
    APIAuthError,
    retry_with_backoff,
    get_circuit_breaker,
    ErrorHandler
)

logger = logging.getLogger("jarvis.api")


class APIManager:
    """
    Manages external API calls with proper error handling.
    
    Features:
    - Automatic retry with exponential backoff
    - Circuit breaker to prevent cascading failures
    - Structured error responses
    """
    
    def __init__(self):
        self.weather_key = os.getenv("OPENWEATHER_API_KEY")
        self.news_key = os.getenv("NEWSAPI_KEY")
        self.stock_key = os.getenv("ALPHAVANTAGE_KEY")

        # Log missing keys but don't fail (graceful degradation)
        missing = []
        if not self.weather_key:
            missing.append("OPENWEATHER_API_KEY")
        if not self.news_key:
            missing.append("NEWSAPI_KEY")
        if not self.stock_key:
            missing.append("ALPHAVANTAGE_KEY")
        
        if missing:
            logger.warning(f"Missing API keys (some features unavailable): {', '.join(missing)}")
        
        # Circuit breakers for each API
        self.weather_breaker = get_circuit_breaker("weather")
        self.news_breaker = get_circuit_breaker("news")
        self.stock_breaker = get_circuit_breaker("stock")

    @retry_with_backoff(max_retries=2, base_delay=1.0)
    async def get_weather(self, city: str) -> str:
        """
        Fetch current weather for `city` from OpenWeatherMap.
        
        Includes:
        - Retry with backoff on transient failures
        - Circuit breaker protection
        - Structured error handling
        """
        if not self.weather_key:
            raise APIAuthError(api_name="Weather", suggestion="Set OPENWEATHER_API_KEY in .env")
        
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={self.weather_key}&units=metric"
        )
        
        async with self.weather_breaker:
            try:
                timeout = aiohttp.ClientTimeout(total=10)
                async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
                    async with session.get(url) as resp:
                        data = await resp.json()
                        
                        if resp.status == 401:
                            raise APIAuthError(api_name="Weather")
                        elif resp.status == 429:
                            raise APIRateLimitError(api_name="Weather", retry_after=60)
                        elif resp.status != 200:
                            msg = data.get("message", f"HTTP {resp.status}")
                            raise APIInvalidResponseError(api_name="Weather", response=msg)
                        
                        desc = data["weather"][0]["description"].title()
                        temp = data["main"]["temp"]
                        hum = data["main"]["humidity"]
                        feels = data["main"].get("feels_like", temp)
                        
                        return (
                            f"🌤️ Weather in {city.title()}:\n"
                            f"• Condition: {desc}\n"
                            f"• Temperature: {temp}°C (feels like {feels}°C)\n"
                            f"• Humidity: {hum}%"
                        )
                        
            except aiohttp.ClientConnectorError as e:
                raise APIConnectionError(api_name="Weather", url=url)
            except asyncio.TimeoutError:
                raise APITimeoutError(api_name="Weather", timeout=10)

    @retry_with_backoff(max_retries=2, base_delay=1.0)
    async def get_news(self, topic: str, page_size: int = 5) -> str:
        """
        Fetch latest news articles for `topic` from NewsAPI.
        
        Includes retry and circuit breaker protection.
        """
        if not self.news_key:
            raise APIAuthError(api_name="News", suggestion="Set NEWSAPI_KEY in .env")
        
        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={topic}&apiKey={self.news_key}&pageSize={page_size}&sortBy=publishedAt"
        )
        
        async with self.news_breaker:
            try:
                timeout = aiohttp.ClientTimeout(total=10)
                async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
                    async with session.get(url) as resp:
                        data = await resp.json()
                        
                        if resp.status == 401:
                            raise APIAuthError(api_name="News")
                        elif resp.status == 429:
                            raise APIRateLimitError(api_name="News", retry_after=60)
                        elif resp.status != 200:
                            msg = data.get("message", f"HTTP {resp.status}")
                            raise APIInvalidResponseError(api_name="News", response=msg)
                        
                        articles = data.get("articles", [])
                        if not articles:
                            return f"ℹ️ No news articles found for '{topic}'."
                        
                        lines = []
                        for art in articles[:page_size]:
                            title = art.get("title", "No title")
                            source = art.get("source", {}).get("name", "Unknown")
                            url_link = art.get("url", "")
                            lines.append(f"• **{title}**\n  Source: {source}\n  {url_link}")
                        
                        return f"📰 Latest news on '{topic}':\n\n" + "\n\n".join(lines)
                        
            except aiohttp.ClientConnectorError:
                raise APIConnectionError(api_name="News")
            except asyncio.TimeoutError:
                raise APITimeoutError(api_name="News", timeout=10)

    @retry_with_backoff(max_retries=2, base_delay=1.0)
    async def get_stock(self, symbol: str) -> str:
        """
        Fetch real-time stock quote for `symbol` from Alpha Vantage.
        
        Includes retry and circuit breaker protection.
        """
        if not self.stock_key:
            raise APIAuthError(api_name="Stock", suggestion="Set ALPHAVANTAGE_KEY in .env")
        
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={self.stock_key}"
        )
        
        async with self.stock_breaker:
            try:
                timeout = aiohttp.ClientTimeout(total=10)
                async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
                    async with session.get(url) as resp:
                        data = await resp.json()
                        
                        # Alpha Vantage returns 200 even on errors
                        if "Error Message" in data:
                            raise APIInvalidResponseError(
                                api_name="Stock",
                                response=data["Error Message"]
                            )
                        
                        if "Note" in data:  # Rate limit message
                            raise APIRateLimitError(api_name="Stock", retry_after=60)
                        
                        quote = data.get("Global Quote", {})
                        price = quote.get("05. price")
                        change = quote.get("09. change")
                        pct = quote.get("10. change percent")
                        
                        if not price:
                            return f"ℹ️ No data found for symbol '{symbol.upper()}'. Check if it's a valid ticker."
                        
                        # Determine trend emoji
                        try:
                            change_val = float(change)
                            trend = "📈" if change_val >= 0 else "📉"
                        except:
                            trend = "💹"
                        
                        return (
                            f"{trend} {symbol.upper()} Stock Quote:\n"
                            f"• Price: ${float(price):.2f}\n"
                            f"• Change: {change} ({pct})"
                        )
                        
            except aiohttp.ClientConnectorError:
                raise APIConnectionError(api_name="Stock")
            except asyncio.TimeoutError:
                raise APITimeoutError(api_name="Stock", timeout=10)
