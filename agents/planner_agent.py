# agents/planner_agent.py

from autogen_agentchat.agents import AssistantAgent

class PlannerAgent(AssistantAgent):
    def __init__(self, name="planner", user_proxy=None, llm_config=None):
        if user_proxy is None:
            raise ValueError("PlannerAgent requires a user_proxy")
        
        super().__init__(
            name=name,
            llm_config=llm_config or {"config_list": [{"model": "gpt-4"}]},
            system_message=(
                "You are the PlannerAgent. Understand user requests and delegate them to TaskAgent, "
                "ToolAgent, or APIAgent,SearchAgent . Respond in the form: 'DELEGATE_TO_[AGENT]: <task>'."
            ),
            user_proxy=user_proxy
        )
        self.agents = {}

    def register_agents(self, agents_dict: dict):
        """Register specialized agents with labels."""
        self.agents = agents_dict

    async def delegate_task(self, user_message: str):
        decision_prompt = (
            f"Given this user request: '{user_message}', determine the most suitable agent: "
            "task, tool, or api. Reply only with the agent label."
        )
        decision = (await self.aask(self.user_proxy, decision_prompt)).strip().lower()

        if decision not in self.agents:
            return f"Planner could not understand the request. No matching agent for: '{decision}'"

        selected = self.agents[decision]
        return await self.aask(selected, user_message)
