# agents/api_agent.py

from autogen_agentchat.agents import AssistantAgent
from .api_manager import APIManager

class APIAgent(AssistantAgent):
    def __init__(self, name="api_agent", model_client=None):
        # ── Updated to match AssistantAgent signature ─────────────────────────────
        super().__init__(
            name=name,
            model_client=model_client,
            system_message=(
                "You are the APIAgent. Fetch live data for weather, news, or stocks. "
                "Use 'weather:<city>', 'news:<topic>' or 'stock:<symbol>' to choose which API to call."
            )
        )
        self.api = APIManager()

    async def get_weather(self, city: str) -> str:
        return await self.api.get_weather(city)

    async def get_news(self, topic: str) -> str:
        return await self.api.get_news(topic)

    async def get_stock(self, symbol: str) -> str:
        return await self.api.get_stock(symbol)

    async def aask(self, user_proxy, message: str, **kwargs) -> str:
        """Route prefixed messages to real APIs, otherwise fallback to LLM."""
        text = message.strip()
        low  = text.lower()

        if low.startswith("weather:"):
            city = text.split(":",1)[1].strip()
            return await self.get_weather(city)

        if low.startswith("news:"):
            topic = text.split(":",1)[1].strip()
            return await self.get_news(topic)

        if low.startswith("stock:"):
            symbol = text.split(":",1)[1].strip()
            return await self.get_stock(symbol)

        # Fallback to standard AssistantAgent LLM behavior
        return await super().aask(user_proxy, message, **kwargs)
