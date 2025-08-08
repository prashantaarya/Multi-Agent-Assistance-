# agents/api_manager.py
import os
import asyncio
import aiohttp

class APIManager:
    def __init__(self):
        self.weather_key = os.getenv("OPENWEATHER_API_KEY")
        self.news_key    = os.getenv("NEWSAPI_KEY")
        self.stock_key   = os.getenv("ALPHAVANTAGE_KEY")

        missing = [name for name, val in {
            "OPENWEATHER_API_KEY": self.weather_key,
            "NEWSAPI_KEY":         self.news_key,
            "ALPHAVANTAGE_KEY":    self.stock_key
        }.items() if not val]
        if missing:
            raise RuntimeError(f"Missing API keys: {', '.join(missing)}")

    async def get_weather(self, city: str) -> str:
        """
        Fetch current weather for `city` from OpenWeatherMap.
        Returns a user-friendly text or error message.
        """
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={self.weather_key}&units=metric"
        )
        try:
            async with aiohttp.ClientSession(trust_env=True) as session:
                async with session.get(url, timeout=10) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        msg = data.get("message", resp.status)
                        return f"❌ Weather error: {msg}"
                    desc = data["weather"][0]["description"].title()
                    temp = data["main"]["temp"]
                    hum  = data["main"]["humidity"]
                    return (
                        f"🌤️ Weather in {city.title()}:\n"
                        f"• {desc}\n"
                        f"• Temperature: {temp}°C\n"
                        f"• Humidity: {hum}%"
                    )
        except aiohttp.ClientConnectorError:
            return "❌ Network error: could not reach the weather service."
        except asyncio.TimeoutError:
            return "❌ Request to weather service timed out."
        except Exception as e:
            return f"❌ Unexpected error fetching weather: {e}"

    async def get_news(self, topic: str, page_size: int = 5) -> str:
        """
        Fetch latest news articles for `topic` from NewsAPI.
        Returns formatted headlines or an error message.
        """
        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={topic}&apiKey={self.news_key}&pageSize={page_size}"
        )
        try:
            async with aiohttp.ClientSession(trust_env=True) as session:
                async with session.get(url, timeout=10) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        msg = data.get("message", resp.status)
                        return f"❌ News error: {msg}"
                    articles = data.get("articles", [])
                    if not articles:
                        return "ℹ️ No news articles found."
                    lines = []
                    for art in articles:
                        title  = art.get("title", "No title")
                        source = art.get("source", {}).get("name", "Unknown")
                        link   = art.get("url", "")
                        lines.append(f"- **{title}** ({source})\n  {link}")
                    return "📰 Latest news:\n" + "\n".join(lines)
        except aiohttp.ClientConnectorError:
            return "❌ Network error: could not reach the news service."
        except asyncio.TimeoutError:
            return "❌ Request to news service timed out."
        except Exception as e:
            return f"❌ Unexpected error fetching news: {e}"

    async def get_stock(self, symbol: str) -> str:
        """
        Fetch real-time stock quote for `symbol` from Alpha Vantage.
        Returns price, change, and percent change or an error message.
        """
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={self.stock_key}"
        )
        try:
            async with aiohttp.ClientSession(trust_env=True) as session:
                async with session.get(url, timeout=10) as resp:
                    data = await resp.json()
                    quote = data.get("Global Quote", {})
                    price = quote.get("05. price")
                    change= quote.get("09. change")
                    pct   = quote.get("10. change percent")
                    if not price:
                        return f"ℹ️ No data found for symbol '{symbol.upper()}'."
                    return (
                        f"💹 {symbol.upper()} Quote:\n"
                        f"• Price: ${price}\n"
                        f"• Change: {change} ({pct})"
                    )
        except aiohttp.ClientConnectorError:
            return "❌ Network error: could not reach the stock service."
        except asyncio.TimeoutError:
            return "❌ Request to stock service timed out."
        except Exception as e:
            return f"❌ Unexpected error fetching stock data: {e}"
