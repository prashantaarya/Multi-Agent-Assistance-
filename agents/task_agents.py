# agents/task_agents.py
from autogen.agentchat import ConversableAgent

class TaskAgent(ConversableAgent):
    def __init__(self, name="task_agent", llm_config=None):
        super().__init__(
            name=name,
            llm_config=llm_config or {"config_list": [{"model": "gpt-4"}]},
            system_message=(
                "You are the TaskAgent. Your job is to handle general task management requests such as reminders, "
                "scheduling, to-do lists, or basic instructions. Keep your replies short and actionable. Do not perform "
                "API calls or execute code. Only confirm and describe how the task would be completed."
            )
        )
