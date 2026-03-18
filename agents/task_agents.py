# agents/task_agents.py
import os
import json
from typing import List
from autogen_agentchat.agents import AssistantAgent
from core.capabilities import register
from core.schemas import ToolParameter, ParameterType

TASKS_FILE = os.getenv("TASKS_FILE", "tasks.json")
PROMPT_V = "task@v1.1"

SYSTEM_MESSAGE = f"""
[{PROMPT_V}]
You are the TaskAgent. You manage a simple to-do list with these commands:
- add task <description>
- list tasks
- complete task <number>
- clear tasks

Always respond with confirmations or the current list. If the command is unclear, ask for a clarification.
""".strip()


class TaskAgent(AssistantAgent):
    """
    Task management agent with industry-standard tool definitions.
    Manages a persistent to-do list stored in JSON.
    """
    
    def __init__(self, name="task", model_client=None):
        super().__init__(name=name, model_client=model_client, system_message=SYSTEM_MESSAGE)
        self.prompt_version = PROMPT_V
        
        # Initialize tasks file
        if not os.path.exists(TASKS_FILE):
            with open(TASKS_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)

        # ──────────────────────────────────────────────────────────────────────
        # Register capabilities with FULL SCHEMAS (Industry Best Practice)
        # ──────────────────────────────────────────────────────────────────────
        
        register(
            capability="todo.add",
            agent_name=self.name,
            handler=self._cap_add,
            description="Add a new task to the to-do list",
            parameters=[
                ToolParameter(
                    name="description",
                    type=ParameterType.STRING,
                    description="The task description to add",
                    required=True
                )
            ],
            category="productivity",
            examples=[
                {"description": "Buy groceries"},
                {"description": "Call mom at 5 PM"},
                {"description": "Review quarterly report"}
            ]
        )
        
        register(
            capability="todo.list",
            agent_name=self.name,
            handler=self._cap_list,
            description="List all current tasks in the to-do list",
            parameters=[],  # No parameters needed
            category="productivity",
            examples=[{}]
        )
        
        register(
            capability="todo.done",
            agent_name=self.name,
            handler=self._cap_complete,
            description="Mark a task as completed by its number",
            parameters=[
                ToolParameter(
                    name="number",
                    type=ParameterType.INTEGER,
                    description="The task number to mark as complete (1-indexed)",
                    required=True
                )
            ],
            category="productivity",
            examples=[
                {"number": 1},
                {"number": 3}
            ]
        )
        
        register(
            capability="todo.clear",
            agent_name=self.name,
            handler=self._cap_clear,
            description="Clear all tasks from the to-do list",
            parameters=[],  # No parameters needed
            category="productivity",
            examples=[{}]
        )

    def _load(self) -> List[str]:
        with open(TASKS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, tasks: List[str]):
        with open(TASKS_FILE, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2)

    # ---- Capabilities -------------------------------------------------------

    async def _cap_add(self, description: str) -> str:
        tasks = self._load()
        tasks.append(description)
        self._save(tasks)
        return f"✅ Added task #{len(tasks)}: {description}"

    async def _cap_list(self) -> str:
        tasks = self._load()
        if not tasks:
            return "🗒️ Your to-do list is empty."
        return "🗒️ To-Do List:\n" + "\n".join(f"{i+1}. {t}" for i, t in enumerate(tasks))

    async def _cap_complete(self, number: int) -> str:
        tasks = self._load()
        idx = number - 1
        if idx < 0 or idx >= len(tasks):
            return f"❌ Invalid task number: {number}."
        done = tasks.pop(idx)
        self._save(tasks)
        return f"✅ Completed task #{number}: {done}"

    async def _cap_clear(self) -> str:
        self._save([])
        return "🗑️ All tasks cleared."

    # ---- Backward-compatible free-text handling ----------------------------

    async def aask(self, user_proxy, message: str, **kwargs) -> str:
        low = (message or "").strip().lower()
        if low.startswith("add task "):
            desc = message[len("add task "):].strip()
            return await self._cap_add(desc)
        if low in ("list tasks", "show tasks"):
            return await self._cap_list()
        if low.startswith("complete task "):
            num = message[len("complete task "):].strip()
            if num.isdigit():
                return await self._cap_complete(int(num))
            return "❌ Use a task number, e.g. 'complete task 2'."
        if low in ("clear tasks", "delete all tasks"):
            return await self._cap_clear()
        return await super().aask(user_proxy, message, **kwargs)
