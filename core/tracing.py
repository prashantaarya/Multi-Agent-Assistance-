# core/tracing.py
"""
Enhanced Tracing System for JARVIS Multi-Agent Assistant

Provides detailed chain-of-thought logging with:
- Visual execution flow
- Agent thinking/reasoning traces
- Capability routing visualization
- Performance metrics
"""

import time
import logging
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

# Configure logger
_LOG = logging.getLogger("jarvis")

# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Agent colors
    PLANNER = "\033[94m"      # Blue
    TASK = "\033[92m"         # Green
    SEARCH = "\033[93m"       # Yellow
    API = "\033[95m"          # Magenta
    TOOL = "\033[96m"         # Cyan
    REACT = "\033[91m"        # Red
    SUPERVISOR = "\033[97m"   # White
    
    # Status colors
    SUCCESS = "\033[92m"      # Green
    ERROR = "\033[91m"        # Red
    THINKING = "\033[93m"     # Yellow
    INFO = "\033[94m"         # Blue


class TraceLevel(Enum):
    MINIMAL = 1      # Just start/end
    NORMAL = 2       # Agent routing + results
    VERBOSE = 3      # Full chain-of-thought
    DEBUG = 4        # Everything including internal state


@dataclass
class ThoughtStep:
    """Represents a single step in the chain of thought"""
    timestamp: datetime
    agent: str
    thought_type: str  # "thinking", "deciding", "acting", "observing"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionTrace:
    """Complete trace of a request execution"""
    request_id: str
    query: str
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    thoughts: List[ThoughtStep] = field(default_factory=list)
    agents_used: List[str] = field(default_factory=list)
    capabilities_called: List[str] = field(default_factory=list)
    total_ms: float = 0.0
    final_output: Optional[str] = None
    
    def add_thought(self, agent: str, thought_type: str, content: str, **metadata):
        """Add a thought step to the trace"""
        self.thoughts.append(ThoughtStep(
            timestamp=datetime.now(),
            agent=agent,
            thought_type=thought_type,
            content=content,
            metadata=metadata
        ))


# Global trace storage (for current request)
_current_trace: Optional[ExecutionTrace] = None
_trace_level: TraceLevel = TraceLevel.VERBOSE


def set_trace_level(level: TraceLevel):
    """Set the global trace verbosity level"""
    global _trace_level
    _trace_level = level


def get_current_trace() -> Optional[ExecutionTrace]:
    """Get the current execution trace"""
    return _current_trace


class EnhancedTracer:
    """
    Enhanced tracer with chain-of-thought logging and visual execution flow.
    
    Features:
    - Visual agent flow (colored output)
    - Chain-of-thought logging
    - Performance metrics
    - Execution trace history
    """
    
    AGENT_COLORS = {
        "planner": Colors.PLANNER,
        "task": Colors.TASK,
        "search": Colors.SEARCH,
        "api": Colors.API,
        "tool": Colors.TOOL,
        "react": Colors.REACT,
        "supervisor": Colors.SUPERVISOR,
    }
    
    AGENT_ICONS = {
        "planner": "🧠",
        "task": "📋",
        "search": "🔍",
        "api": "🌐",
        "tool": "🔧",
        "react": "🔄",
        "supervisor": "👁️",
    }
    
    def __init__(self, request_id: str | None):
        global _current_trace
        self.rid = request_id or self._generate_short_id()
        self.indent_level = 0
        self.start_time = time.perf_counter()
        
        # Create new execution trace
        _current_trace = ExecutionTrace(request_id=self.rid, query="")
        self.trace = _current_trace
    
    def _generate_short_id(self) -> str:
        """Generate a short readable request ID"""
        import random
        return f"{random.randint(1000, 9999)}"
    
    def _get_agent_color(self, agent: str) -> str:
        """Get color code for an agent"""
        return self.AGENT_COLORS.get(agent.lower(), Colors.INFO)
    
    def _get_agent_icon(self, agent: str) -> str:
        """Get icon for an agent"""
        return self.AGENT_ICONS.get(agent.lower(), "🤖")
    
    def _format_indent(self) -> str:
        """Get indentation string"""
        return "  " * self.indent_level
    
    def _log(self, message: str, level: TraceLevel = TraceLevel.NORMAL):
        """Log message if trace level allows"""
        if _trace_level.value >= level.value:
            _LOG.info(message)
    
    # ──────────────────────────────────────────────────────────────────────────
    # Chain of Thought Methods
    # ──────────────────────────────────────────────────────────────────────────
    
    def thought(self, agent: str, content: str, **metadata):
        """Log an agent's thinking process"""
        icon = self._get_agent_icon(agent)
        color = self._get_agent_color(agent)
        indent = self._format_indent()
        
        self._log(
            f"{indent}{color}{icon} [{agent.upper()}] 💭 Thinking: {content}{Colors.RESET}",
            TraceLevel.VERBOSE
        )
        self.trace.add_thought(agent, "thinking", content, **metadata)
    
    def decision(self, agent: str, decision: str, confidence: float = None, **metadata):
        """Log an agent's decision"""
        icon = self._get_agent_icon(agent)
        color = self._get_agent_color(agent)
        indent = self._format_indent()
        
        conf_str = f" (confidence: {confidence:.0%})" if confidence else ""
        self._log(
            f"{indent}{color}{icon} [{agent.upper()}] ✅ Decision: {decision}{conf_str}{Colors.RESET}",
            TraceLevel.NORMAL
        )
        self.trace.add_thought(agent, "deciding", decision, confidence=confidence, **metadata)
    
    def action(self, agent: str, capability: str, inputs: dict = None):
        """Log an agent taking an action"""
        icon = self._get_agent_icon(agent)
        color = self._get_agent_color(agent)
        indent = self._format_indent()
        
        inputs_str = f" with {inputs}" if inputs else ""
        self._log(
            f"{indent}{color}{icon} [{agent.upper()}] ⚡ Action: {capability}{inputs_str}{Colors.RESET}",
            TraceLevel.NORMAL
        )
        self.trace.add_thought(agent, "acting", f"{capability}{inputs_str}", capability=capability, inputs=inputs)
        self.trace.capabilities_called.append(capability)
    
    def observation(self, agent: str, result: str, truncate: int = 100):
        """Log an agent's observation/result"""
        icon = self._get_agent_icon(agent)
        color = self._get_agent_color(agent)
        indent = self._format_indent()
        
        # Truncate long results
        display_result = result[:truncate] + "..." if len(result) > truncate else result
        display_result = display_result.replace("\n", " ")
        
        self._log(
            f"{indent}{color}{icon} [{agent.upper()}] 👁️ Result: {display_result}{Colors.RESET}",
            TraceLevel.NORMAL
        )
        self.trace.add_thought(agent, "observing", result)
    
    # ──────────────────────────────────────────────────────────────────────────
    # Flow Control Methods
    # ──────────────────────────────────────────────────────────────────────────
    
    def start_request(self, query: str):
        """Mark the start of a new request"""
        self.trace.query = query
        self._log(
            f"\n{'='*60}\n"
            f"{Colors.BOLD}🚀 NEW REQUEST [rid={self.rid}]{Colors.RESET}\n"
            f"{Colors.DIM}Query: {query[:100]}{'...' if len(query) > 100 else ''}{Colors.RESET}\n"
            f"{'='*60}",
            TraceLevel.MINIMAL
        )
    
    def end_request(self, output: str = None):
        """Mark the end of a request"""
        self.trace.completed_at = datetime.now()
        self.trace.total_ms = (time.perf_counter() - self.start_time) * 1000
        self.trace.final_output = output
        
        agents_str = " → ".join(self.trace.agents_used) if self.trace.agents_used else "none"
        
        self._log(
            f"\n{Colors.SUCCESS}✅ REQUEST COMPLETE [rid={self.rid}]{Colors.RESET}\n"
            f"{Colors.DIM}Agents: {agents_str}{Colors.RESET}\n"
            f"{Colors.DIM}Total time: {self.trace.total_ms:.1f}ms{Colors.RESET}\n"
            f"{'='*60}\n",
            TraceLevel.MINIMAL
        )
    
    def enter_agent(self, agent: str, task: str = None):
        """Log entering an agent's execution"""
        icon = self._get_agent_icon(agent)
        color = self._get_agent_color(agent)
        indent = self._format_indent()
        
        task_str = f": {task[:60]}..." if task and len(task) > 60 else (f": {task}" if task else "")
        
        self._log(
            f"{indent}{color}┌─ {icon} ENTER [{agent.upper()}]{task_str}{Colors.RESET}",
            TraceLevel.NORMAL
        )
        self.indent_level += 1
        self.trace.agents_used.append(agent)
    
    def exit_agent(self, agent: str, duration_ms: float = None):
        """Log exiting an agent's execution"""
        self.indent_level = max(0, self.indent_level - 1)
        icon = self._get_agent_icon(agent)
        color = self._get_agent_color(agent)
        indent = self._format_indent()
        
        time_str = f" ({duration_ms:.1f}ms)" if duration_ms else ""
        
        self._log(
            f"{indent}{color}└─ {icon} EXIT [{agent.upper()}]{time_str}{Colors.RESET}",
            TraceLevel.NORMAL
        )
    
    def route(self, from_agent: str, to_agent: str, reason: str = None):
        """Log routing from one agent to another"""
        from_icon = self._get_agent_icon(from_agent)
        to_icon = self._get_agent_icon(to_agent)
        indent = self._format_indent()
        
        reason_str = f" ({reason})" if reason else ""
        
        self._log(
            f"{indent}{Colors.INFO}{from_icon} [{from_agent.upper()}] ──▶ {to_icon} [{to_agent.upper()}]{reason_str}{Colors.RESET}",
            TraceLevel.NORMAL
        )
    
    # ──────────────────────────────────────────────────────────────────────────
    # Backward Compatible Span (with enhancements)
    # ──────────────────────────────────────────────────────────────────────────
    
    @contextmanager
    def span(self, name: str, **fields):
        """
        Context manager for timing operations (backward compatible).
        Enhanced with better visual output.
        """
        t0 = time.perf_counter()
        
        # Parse span name for better display
        parts = name.split(".")
        agent = parts[0] if parts else "system"
        operation = parts[1] if len(parts) > 1 else name
        
        # Determine appropriate logging based on operation
        if operation == "call":
            self.enter_agent(agent, fields.get("message_preview"))
        elif operation == "parse":
            self.thought(agent, "Parsing response...")
        elif operation == "resolve":
            cap = fields.get("capability", "unknown")
            conf = fields.get("confidence")
            self.decision(agent, f"Route to {cap}", confidence=conf)
        elif operation == "exec":
            cap = fields.get("capability", "unknown")
            agent_name = fields.get("agent", agent)
            self.action(agent_name, cap)
        
        try:
            yield
        finally:
            dt = (time.perf_counter() - t0) * 1000
            
            if operation == "call":
                self.exit_agent(agent, dt)
            elif operation == "exec":
                # Log completion time for executions
                self._log(
                    f"{self._format_indent()}{Colors.DIM}   ⏱️ Completed in {dt:.1f}ms{Colors.RESET}",
                    TraceLevel.VERBOSE
                )


# ──────────────────────────────────────────────────────────────────────────────
# Factory function (backward compatible)
# ──────────────────────────────────────────────────────────────────────────────

def get_tracer(request_id: str | None) -> EnhancedTracer:
    """Get a tracer instance for the given request ID"""
    return EnhancedTracer(request_id)


# Legacy alias
Tracer = EnhancedTracer
