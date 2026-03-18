# core/errors.py
"""
Structured Error Handling System for JARVIS Multi-Agent Assistant

Industry best practices implemented:
- Hierarchical error types
- Retry with exponential backoff
- Circuit breaker pattern
- Structured error context
- User-friendly messages

Usage:
    from core.errors import ToolExecutionError, retry_with_backoff

    @retry_with_backoff(max_retries=3)
    async def call_api():
        ...

    raise ToolExecutionError(
        message="Weather API failed",
        details={"city": "NYC"},
        suggestion="Try again later"
    )
"""

import asyncio
import time
import logging
import functools
from typing import Optional, Dict, Any, Callable, Type
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger("jarvis.errors")


# ──────────────────────────────────────────────────────────────────────────────
# Error Codes (for programmatic handling)
# ──────────────────────────────────────────────────────────────────────────────

class ErrorCode(str, Enum):
    """Standardized error codes for all JARVIS errors"""
    
    # Planner errors (1xxx)
    INTENT_NOT_UNDERSTOOD = "E1001"
    LOW_CONFIDENCE = "E1002"
    PARSING_ERROR = "E1003"
    ROUTING_FAILED = "E1004"
    
    # Capability errors (2xxx)
    CAPABILITY_NOT_FOUND = "E2001"
    CAPABILITY_TIMEOUT = "E2002"
    CAPABILITY_EXECUTION_FAILED = "E2003"
    CAPABILITY_INVALID_INPUT = "E2004"
    
    # Agent errors (3xxx)
    AGENT_NOT_FOUND = "E3001"
    AGENT_TIMEOUT = "E3002"
    AGENT_COMMUNICATION_FAILED = "E3003"
    AGENT_BUSY = "E3004"
    
    # Tool errors (4xxx)
    TOOL_NOT_FOUND = "E4001"
    TOOL_INPUT_VALIDATION = "E4002"
    TOOL_EXECUTION_FAILED = "E4003"
    TOOL_TIMEOUT = "E4004"
    TOOL_PERMISSION_DENIED = "E4005"
    
    # External API errors (5xxx)
    API_CONNECTION_FAILED = "E5001"
    API_TIMEOUT = "E5002"
    API_RATE_LIMITED = "E5003"
    API_AUTH_FAILED = "E5004"
    API_INVALID_RESPONSE = "E5005"
    API_SERVICE_UNAVAILABLE = "E5006"
    
    # ReAct errors (6xxx)
    REACT_MAX_ITERATIONS = "E6001"
    REACT_STUCK_LOOP = "E6002"
    REACT_NO_PROGRESS = "E6003"
    
    # General errors (9xxx)
    UNKNOWN_ERROR = "E9001"
    CONFIGURATION_ERROR = "E9002"
    RESOURCE_EXHAUSTED = "E9003"


# ──────────────────────────────────────────────────────────────────────────────
# Base Error Class
# ──────────────────────────────────────────────────────────────────────────────

class JARVISError(Exception):
    """
    Base exception class for all JARVIS errors.
    
    Attributes:
        code: Unique error code for programmatic handling
        message: Technical message for developers
        user_message: Safe message to show to users
        recoverable: Whether the error can be retried
        retry_after: Seconds to wait before retry (if recoverable)
        fallback: Alternative capability to try
        details: Additional context about the error
        suggestion: How to fix or work around the error
        timestamp: When the error occurred
    """
    
    def __init__(
        self,
        code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        message: str = "An error occurred",
        user_message: str = None,
        recoverable: bool = False,
        retry_after: int = None,
        fallback: str = None,
        details: Dict[str, Any] = None,
        suggestion: str = None
    ):
        self.code = code
        self.message = message
        self.user_message = user_message or self._default_user_message()
        self.recoverable = recoverable
        self.retry_after = retry_after
        self.fallback = fallback
        self.details = details or {}
        self.suggestion = suggestion
        self.timestamp = datetime.utcnow()
        
        super().__init__(self.message)
    
    def _default_user_message(self) -> str:
        """Generate a user-friendly message"""
        return "I encountered an issue processing your request. Please try again."
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for API responses"""
        return {
            "error": {
                "code": self.code.value,
                "message": self.user_message,
                "recoverable": self.recoverable,
                "retry_after": self.retry_after,
                "fallback": self.fallback,
                "suggestion": self.suggestion,
                "timestamp": self.timestamp.isoformat()
            }
        }
    
    def __str__(self):
        return f"[{self.code.value}] {self.message}"
    
    def __repr__(self):
        return f"{self.__class__.__name__}(code={self.code}, message={self.message!r})"


# ──────────────────────────────────────────────────────────────────────────────
# Planner Errors
# ──────────────────────────────────────────────────────────────────────────────

class PlannerError(JARVISError):
    """Base class for planner-related errors"""
    pass


class IntentNotUnderstoodError(PlannerError):
    """Raised when planner cannot understand user intent"""
    
    def __init__(self, query: str, **kwargs):
        super().__init__(
            code=ErrorCode.INTENT_NOT_UNDERSTOOD,
            message=f"Could not understand intent for: {query[:100]}",
            user_message="I'm not sure what you're asking. Could you rephrase that?",
            recoverable=True,
            suggestion="Try rephrasing your question or being more specific",
            details={"query": query},
            **kwargs
        )


class LowConfidenceError(PlannerError):
    """Raised when planner has low confidence in its decision"""
    
    def __init__(self, confidence: float, capability: str, **kwargs):
        super().__init__(
            code=ErrorCode.LOW_CONFIDENCE,
            message=f"Low confidence ({confidence:.0%}) for capability: {capability}",
            user_message="I'm not fully confident about this. Let me try a different approach.",
            recoverable=True,
            fallback="search.web",  # Default fallback
            details={"confidence": confidence, "capability": capability},
            **kwargs
        )


class ParsingError(PlannerError):
    """Raised when planner output cannot be parsed"""
    
    def __init__(self, raw_output: str, **kwargs):
        super().__init__(
            code=ErrorCode.PARSING_ERROR,
            message=f"Failed to parse planner output",
            user_message="I had trouble processing that. Let me try again.",
            recoverable=True,
            details={"raw_output": raw_output[:500]},
            **kwargs
        )


# ──────────────────────────────────────────────────────────────────────────────
# Capability Errors
# ──────────────────────────────────────────────────────────────────────────────

class CapabilityError(JARVISError):
    """Base class for capability-related errors"""
    pass


class CapabilityNotFoundError(CapabilityError):
    """Raised when a capability is not registered"""
    
    def __init__(self, capability: str, available: list = None, **kwargs):
        super().__init__(
            code=ErrorCode.CAPABILITY_NOT_FOUND,
            message=f"Capability not found: {capability}",
            user_message="I don't have that ability yet.",
            recoverable=False,
            details={"capability": capability, "available": available or []},
            suggestion=f"Available capabilities: {', '.join(available or [])}",
            **kwargs
        )


class CapabilityTimeoutError(CapabilityError):
    """Raised when a capability times out"""
    
    def __init__(self, capability: str, timeout: float, **kwargs):
        super().__init__(
            code=ErrorCode.CAPABILITY_TIMEOUT,
            message=f"Capability {capability} timed out after {timeout}s",
            user_message="That's taking too long. Let me try a simpler approach.",
            recoverable=True,
            retry_after=5,
            details={"capability": capability, "timeout": timeout},
            **kwargs
        )


class CapabilityExecutionError(CapabilityError):
    """Raised when a capability fails to execute"""
    
    def __init__(self, capability: str, error: str, **kwargs):
        super().__init__(
            code=ErrorCode.CAPABILITY_EXECUTION_FAILED,
            message=f"Capability {capability} failed: {error}",
            user_message="Something went wrong with that action. Let me try differently.",
            recoverable=True,
            details={"capability": capability, "error": error},
            **kwargs
        )


# ──────────────────────────────────────────────────────────────────────────────
# Agent Errors
# ──────────────────────────────────────────────────────────────────────────────

class AgentError(JARVISError):
    """Base class for agent-related errors"""
    pass


class AgentNotFoundError(AgentError):
    """Raised when an agent is not found"""
    
    def __init__(self, agent_name: str, available: list = None, **kwargs):
        super().__init__(
            code=ErrorCode.AGENT_NOT_FOUND,
            message=f"Agent not found: {agent_name}",
            user_message="That specialist isn't available right now.",
            recoverable=False,
            details={"agent": agent_name, "available": available or []},
            **kwargs
        )


class AgentTimeoutError(AgentError):
    """Raised when an agent takes too long to respond"""
    
    def __init__(self, agent_name: str, timeout: float, **kwargs):
        super().__init__(
            code=ErrorCode.AGENT_TIMEOUT,
            message=f"Agent {agent_name} timed out after {timeout}s",
            user_message="My specialist is taking too long. Let me try another approach.",
            recoverable=True,
            retry_after=3,
            details={"agent": agent_name, "timeout": timeout},
            **kwargs
        )


class AgentCommunicationError(AgentError):
    """Raised when agents fail to communicate"""
    
    def __init__(self, from_agent: str, to_agent: str, error: str, **kwargs):
        super().__init__(
            code=ErrorCode.AGENT_COMMUNICATION_FAILED,
            message=f"Communication failed: {from_agent} -> {to_agent}: {error}",
            user_message="My team had a communication issue. Let me handle this directly.",
            recoverable=True,
            details={"from": from_agent, "to": to_agent, "error": error},
            **kwargs
        )


# ──────────────────────────────────────────────────────────────────────────────
# Tool Errors
# ──────────────────────────────────────────────────────────────────────────────

class ToolError(JARVISError):
    """Base class for tool-related errors"""
    pass


class ToolNotFoundError(ToolError):
    """Raised when a tool is not found"""
    
    def __init__(self, tool_name: str, **kwargs):
        super().__init__(
            code=ErrorCode.TOOL_NOT_FOUND,
            message=f"Tool not found: {tool_name}",
            user_message="I don't have access to that tool.",
            recoverable=False,
            details={"tool": tool_name},
            **kwargs
        )


class ToolInputValidationError(ToolError):
    """Raised when tool inputs are invalid"""
    
    def __init__(self, tool_name: str, param: str, error: str, **kwargs):
        super().__init__(
            code=ErrorCode.TOOL_INPUT_VALIDATION,
            message=f"Invalid input for {tool_name}.{param}: {error}",
            user_message="I need more information to do that. Could you be more specific?",
            recoverable=True,
            details={"tool": tool_name, "parameter": param, "error": error},
            suggestion=f"Please provide a valid value for {param}",
            **kwargs
        )


class ToolExecutionError(ToolError):
    """Raised when a tool fails to execute"""
    
    def __init__(self, tool_name: str, error: str, **kwargs):
        super().__init__(
            code=ErrorCode.TOOL_EXECUTION_FAILED,
            message=f"Tool {tool_name} failed: {error}",
            user_message="That action didn't work. Let me try something else.",
            recoverable=True,
            details={"tool": tool_name, "error": error},
            **kwargs
        )


class ToolTimeoutError(ToolError):
    """Raised when a tool times out"""
    
    def __init__(self, tool_name: str, timeout: float, **kwargs):
        super().__init__(
            code=ErrorCode.TOOL_TIMEOUT,
            message=f"Tool {tool_name} timed out after {timeout}s",
            user_message="That's taking too long. Let me try again.",
            recoverable=True,
            retry_after=2,
            details={"tool": tool_name, "timeout": timeout},
            **kwargs
        )


# ──────────────────────────────────────────────────────────────────────────────
# External API Errors
# ──────────────────────────────────────────────────────────────────────────────

class ExternalAPIError(JARVISError):
    """Base class for external API errors"""
    pass


class APIConnectionError(ExternalAPIError):
    """Raised when API connection fails"""
    
    def __init__(self, api_name: str, url: str = None, **kwargs):
        super().__init__(
            code=ErrorCode.API_CONNECTION_FAILED,
            message=f"Failed to connect to {api_name}",
            user_message=f"I couldn't reach the {api_name} service. It might be down.",
            recoverable=True,
            retry_after=5,
            details={"api": api_name, "url": url},
            **kwargs
        )


class APITimeoutError(ExternalAPIError):
    """Raised when API request times out"""
    
    def __init__(self, api_name: str, timeout: float, **kwargs):
        super().__init__(
            code=ErrorCode.API_TIMEOUT,
            message=f"{api_name} API timed out after {timeout}s",
            user_message=f"The {api_name} service is slow right now. Try again shortly.",
            recoverable=True,
            retry_after=3,
            details={"api": api_name, "timeout": timeout},
            **kwargs
        )


class APIRateLimitError(ExternalAPIError):
    """Raised when API rate limit is hit"""
    
    def __init__(self, api_name: str, retry_after: int = 60, **kwargs):
        super().__init__(
            code=ErrorCode.API_RATE_LIMITED,
            message=f"{api_name} rate limit exceeded",
            user_message=f"I've made too many requests to {api_name}. Please wait a moment.",
            recoverable=True,
            retry_after=retry_after,
            details={"api": api_name},
            **kwargs
        )


class APIAuthError(ExternalAPIError):
    """Raised when API authentication fails"""
    
    def __init__(self, api_name: str, **kwargs):
        super().__init__(
            code=ErrorCode.API_AUTH_FAILED,
            message=f"{api_name} authentication failed",
            user_message=f"I'm having trouble accessing {api_name}. Please check my configuration.",
            recoverable=False,
            details={"api": api_name},
            suggestion="Check API key configuration",
            **kwargs
        )


class APIInvalidResponseError(ExternalAPIError):
    """Raised when API returns unexpected response"""
    
    def __init__(self, api_name: str, response: str = None, **kwargs):
        super().__init__(
            code=ErrorCode.API_INVALID_RESPONSE,
            message=f"{api_name} returned invalid response",
            user_message=f"I got an unexpected response from {api_name}.",
            recoverable=True,
            details={"api": api_name, "response": response[:200] if response else None},
            **kwargs
        )


# ──────────────────────────────────────────────────────────────────────────────
# ReAct Errors
# ──────────────────────────────────────────────────────────────────────────────

class ReActError(JARVISError):
    """Base class for ReAct-related errors"""
    pass


class ReActMaxIterationsError(ReActError):
    """Raised when ReAct hits max iterations"""
    
    def __init__(self, iterations: int, partial_result: str = None, **kwargs):
        super().__init__(
            code=ErrorCode.REACT_MAX_ITERATIONS,
            message=f"ReAct reached max iterations ({iterations})",
            user_message="This is taking longer than expected. Here's what I found so far.",
            recoverable=False,
            details={"iterations": iterations, "partial_result": partial_result},
            **kwargs
        )


class ReActStuckError(ReActError):
    """Raised when ReAct appears to be stuck in a loop"""
    
    def __init__(self, repeated_action: str, **kwargs):
        super().__init__(
            code=ErrorCode.REACT_STUCK_LOOP,
            message=f"ReAct stuck repeating: {repeated_action}",
            user_message="I seem to be going in circles. Let me try a different approach.",
            recoverable=True,
            fallback="planner",
            details={"repeated_action": repeated_action},
            **kwargs
        )


# ──────────────────────────────────────────────────────────────────────────────
# Retry Decorator with Exponential Backoff
# ──────────────────────────────────────────────────────────────────────────────

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    retryable_errors: tuple = (ExternalAPIError, ToolTimeoutError, AgentTimeoutError),
    on_retry: Callable = None
):
    """
    Decorator for retrying async functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential backoff calculation
        retryable_errors: Tuple of exception types that should trigger retry
        on_retry: Optional callback function(attempt, error, delay)
    
    Example:
        @retry_with_backoff(max_retries=3)
        async def fetch_weather(city: str):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_errors as e:
                    last_error = e
                    
                    # Check if error says it's not recoverable
                    if isinstance(e, JARVISError) and not e.recoverable:
                        raise
                    
                    if attempt == max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}")
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )
                    
                    # Use retry_after from error if available
                    if isinstance(e, JARVISError) and e.retry_after:
                        delay = max(delay, e.retry_after)
                    
                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                        f"after {delay:.1f}s due to: {e}"
                    )
                    
                    if on_retry:
                        on_retry(attempt + 1, e, delay)
                    
                    await asyncio.sleep(delay)
                except Exception as e:
                    # Non-retryable error
                    raise
            
            raise last_error
        return wrapper
    return decorator


# ──────────────────────────────────────────────────────────────────────────────
# Circuit Breaker Pattern
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CircuitBreakerState:
    """State for a circuit breaker"""
    failures: int = 0
    last_failure: Optional[datetime] = None
    state: str = "closed"  # closed, open, half-open
    success_count: int = 0


class CircuitBreaker:
    """
    Circuit breaker pattern for external API calls.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests fail immediately
    - HALF-OPEN: Testing if service recovered
    
    Example:
        weather_breaker = CircuitBreaker("weather_api", failure_threshold=5)
        
        async def get_weather(city):
            async with weather_breaker:
                return await call_weather_api(city)
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        half_open_requests: int = 3
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests
        self._state = CircuitBreakerState()
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)"""
        if self._state.state == "open":
            # Check if recovery timeout has passed
            if self._state.last_failure:
                elapsed = (datetime.utcnow() - self._state.last_failure).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self._state.state = "half-open"
                    self._state.success_count = 0
                    logger.info(f"Circuit {self.name} entering half-open state")
                    return False
            return True
        return False
    
    def record_success(self):
        """Record a successful call"""
        if self._state.state == "half-open":
            self._state.success_count += 1
            if self._state.success_count >= self.half_open_requests:
                self._state.state = "closed"
                self._state.failures = 0
                logger.info(f"Circuit {self.name} closed after recovery")
        else:
            self._state.failures = 0
    
    def record_failure(self, error: Exception = None):
        """Record a failed call"""
        self._state.failures += 1
        self._state.last_failure = datetime.utcnow()
        
        if self._state.state == "half-open":
            self._state.state = "open"
            logger.warning(f"Circuit {self.name} re-opened after half-open failure")
        elif self._state.failures >= self.failure_threshold:
            self._state.state = "open"
            logger.warning(f"Circuit {self.name} opened after {self._state.failures} failures")
    
    async def __aenter__(self):
        """Async context manager entry"""
        if self.is_open:
            raise APIConnectionError(
                api_name=self.name,
                recoverable=True,
                retry_after=self.recovery_timeout
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if exc_type is None:
            self.record_success()
        else:
            self.record_failure(exc_val)
        return False  # Don't suppress exceptions


# ──────────────────────────────────────────────────────────────────────────────
# Global Circuit Breakers for External APIs
# ──────────────────────────────────────────────────────────────────────────────

circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(api_name: str) -> CircuitBreaker:
    """Get or create a circuit breaker for an API"""
    if api_name not in circuit_breakers:
        circuit_breakers[api_name] = CircuitBreaker(api_name)
    return circuit_breakers[api_name]


# ──────────────────────────────────────────────────────────────────────────────
# Error Handler Utility
# ──────────────────────────────────────────────────────────────────────────────

class ErrorHandler:
    """
    Utility class for handling errors consistently across the application.
    """
    
    @staticmethod
    def handle(error: Exception, context: str = None) -> JARVISError:
        """
        Convert any exception to a JARVISError with proper context.
        """
        if isinstance(error, JARVISError):
            return error
        
        # Map common exceptions to JARVIS errors
        if isinstance(error, asyncio.TimeoutError):
            return ToolTimeoutError(
                tool_name=context or "unknown",
                timeout=30.0
            )
        elif isinstance(error, ConnectionError):
            return APIConnectionError(
                api_name=context or "unknown"
            )
        elif isinstance(error, ValueError):
            return ToolInputValidationError(
                tool_name=context or "unknown",
                param="input",
                error=str(error)
            )
        else:
            return JARVISError(
                code=ErrorCode.UNKNOWN_ERROR,
                message=str(error),
                details={"original_type": type(error).__name__, "context": context}
            )
    
    @staticmethod
    def format_for_user(error: JARVISError) -> str:
        """Format error message for display to user"""
        message = error.user_message
        if error.suggestion:
            message += f"\n💡 Suggestion: {error.suggestion}"
        if error.recoverable and error.retry_after:
            message += f"\n⏱️ Try again in {error.retry_after} seconds"
        return message
    
    @staticmethod
    def format_for_log(error: JARVISError) -> str:
        """Format error for logging"""
        return (
            f"[{error.code.value}] {error.message} | "
            f"recoverable={error.recoverable} | "
            f"details={error.details}"
        )
