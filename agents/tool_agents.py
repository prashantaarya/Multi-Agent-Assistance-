# agents/tool_agents.py
from autogen_agentchat.agents import AssistantAgent
from .docker_manager import DockerManager
from core.capabilities import register
from core.schemas import ToolParameter, ParameterType

PROMPT_V = "tool@v1.1"
SYSTEM_MESSAGE = f"""
[{PROMPT_V}]
You are the ToolAgent. You execute Python code safely in a sandbox and return stdout/stderr.
If instructions are unclear or unsafe, ask for clarification.
""".strip()


class ToolAgent(AssistantAgent):
    """
    Code execution agent with sandboxed Docker environment.
    Safely executes Python code and returns results.
    """
    
    def __init__(self, name="tool", model_client=None):
        super().__init__(name=name, model_client=model_client, system_message=SYSTEM_MESSAGE)
        self.prompt_version = PROMPT_V
        self.docker = DockerManager()
        
        # ──────────────────────────────────────────────────────────────────────
        # Register capability with FULL SCHEMA (Industry Best Practice)
        # ──────────────────────────────────────────────────────────────────────
        
        register(
            capability="code.execute",
            agent_name=self.name,
            handler=self._cap_exec,
            description="Execute Python code safely in a sandboxed Docker environment and return the output",
            parameters=[
                ToolParameter(
                    name="code_snippet",
                    type=ParameterType.STRING,
                    description="The Python code to execute. Can be single or multi-line code.",
                    required=True
                )
            ],
            category="code",
            examples=[
                {"code_snippet": "print(2 + 2)"},
                {"code_snippet": "import math\nprint(math.sqrt(16))"},
                {"code_snippet": "for i in range(5): print(i)"}
            ]
        )

    async def _cap_exec(self, code_snippet: str) -> str:
        output = self.docker.run_code(code_snippet)
        return f"```python\n{output}\n```"

    async def aask(self, user_proxy, message: str, **kwargs) -> str:
        # optional: parse "run: <code>"
        low = message.strip().lower()
        if low.startswith("run:"):
            return await self._cap_exec(message.split(":",1)[1].strip())
        return await super().aask(user_proxy, message, **kwargs)
