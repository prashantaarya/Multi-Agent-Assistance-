# agents/tool_agents.py
from autogen.agentchat import ConversableAgent

class ToolAgent(ConversableAgent):
    def __init__(self, name="tool_agent", llm_config=None):
        super().__init__(
            name=name,
            llm_config=llm_config or {"config_list": [{"model": "gpt-4"}]},
            system_message=(
                "You are the ToolAgent. Your responsibility is to solve problems using tools like code execution, "
                "calculations, and text/file processing. When necessary, provide step-by-step output of your logic. "
                "Do not handle general task reminders or call APIs. Stay in your domain."
            )
        )
        
        
        self.docker = DockerManager()

    async def execute_python(self, code_snippet: str) -> str:
        """
        Execute a Python snippet inside a sandboxed Docker container.
        Returns stdout/stderr wrapped in a Markdown code block.
        """
        output = self.docker.run_code(code_snippet)
        return f"```python\n{output}\n```"
