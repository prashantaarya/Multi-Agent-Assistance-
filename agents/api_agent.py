# agents/api_agent.py

from typing import Dict, Any
from autogen_agentchat.agents import AssistantAgent
from .api_manager import APIManager
from core.capabilities import register
from core.schemas import ToolParameter, ParameterType


class APIAgent(AssistantAgent):
    """
    External API agent for fetching live data.
    Handles weather, news, and stock information.
    """
    
    def __init__(self, name="api_agent", model_client=None):
        super().__init__(
            name=name,
            model_client=model_client,
            system_message=(
                "You are the APIAgent. Fetch live data for weather, news, or stocks. "
                "Use 'weather:<city>', 'news:<topic>' or 'stock:<symbol>' to choose which API to call."
            )
        )
        self.api = APIManager()

        # ──────────────────────────────────────────────────────────────────────
        # Register capabilities with FULL SCHEMAS (Industry Best Practice)
        # ──────────────────────────────────────────────────────────────────────
        
        register(
            capability="weather.read",
            agent_name=self.name,
            handler=self.get_weather,
            description="Get current weather information for a specified city",
            parameters=[
                ToolParameter(
                    name="city",
                    type=ParameterType.STRING,
                    description="The city name to get weather for (e.g., 'Hyderabad', 'New York')",
                    required=True
                )
            ],
            category="api",
            examples=[
                {"city": "Hyderabad"},
                {"city": "London"},
                {"city": "New York"}
            ]
        )
        
        register(
            capability="news.read",
            agent_name=self.name,
            handler=self.get_news,
            description="Get latest news articles on a specified topic",
            parameters=[
                ToolParameter(
                    name="topic",
                    type=ParameterType.STRING,
                    description="The news topic to search for (e.g., 'AI', 'technology', 'sports')",
                    required=True
                )
            ],
            category="api",
            examples=[
                {"topic": "artificial intelligence"},
                {"topic": "cricket"},
                {"topic": "stock market"}
            ]
        )
        
        register(
            capability="stock.read",
            agent_name=self.name,
            handler=self.get_stock,
            description="Get current stock price and information for a ticker symbol",
            parameters=[
                ToolParameter(
                    name="symbol",
                    type=ParameterType.STRING,
                    description="The stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')",
                    required=True
                )
            ],
            category="api",
            examples=[
                {"symbol": "AAPL"},
                {"symbol": "TSLA"},
                {"symbol": "RELIANCE.NS"}
            ]
        )

    async def get_weather(self, city: str) -> Dict[str, Any]:
        """Get weather with error handling and structured data"""
        try:
            return await self.api.get_weather(city)
        except Exception as e:
            from core.errors import JARVISError, ErrorHandler
            if isinstance(e, JARVISError):
                error_msg = ErrorHandler.format_for_user(e)
            else:
                error_msg = f"❌ Weather error: {str(e)}"
            return {"response": error_msg, "data": None}

    async def get_news(self, topic: str) -> Dict[str, Any]:
        """Get news with error handling and structured data"""
        try:
            return await self.api.get_news(topic)
        except Exception as e:
            from core.errors import JARVISError, ErrorHandler
            if isinstance(e, JARVISError):
                error_msg = ErrorHandler.format_for_user(e)
            else:
                error_msg = f"❌ News error: {str(e)}"
            return {"response": error_msg, "data": None}

    async def get_stock(self, symbol: str) -> Dict[str, Any]:
        """Get stock with error handling and structured data"""
        try:
            return await self.api.get_stock(symbol)
        except Exception as e:
            from core.errors import JARVISError, ErrorHandler
            if isinstance(e, JARVISError):
                error_msg = ErrorHandler.format_for_user(e)
            else:
                error_msg = f"❌ Stock error: {str(e)}"
            return {"response": error_msg, "data": None}

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
