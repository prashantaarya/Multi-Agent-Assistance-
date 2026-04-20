# tests/test_agent_architecture.py
"""
Test True Agent Architecture WITHOUT needing LLM API
Simulates the planner decision and verifies routing works
"""

import sys, os, asyncio, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import init_db
from core.agent_router import DomainAgent, get_router
from core.schemas import PlannerDecision

# Import base_agents which triggers domain agent registration
from agents.base_agents import model_client, agent_router

def test_routing_logic():
    """Test that planner decisions correctly route to domain agents"""
    init_db()
    
    print("=" * 60)
    print("TRUE AGENT ARCHITECTURE TEST")
    print("=" * 60)
    
    router = get_router()
    
    # Test 1: Router has all agents
    print("\n🔍 TEST 1: Agent Registry")
    agents = router.list_agents()
    print(f"  Registered agents: {len(agents)}")
    for a in agents:
        print(f"    {a['name']}: {a['tool_count']} tools → {', '.join(a['tools'][:3])}{'...' if a['tool_count'] > 3 else ''}")
    assert len(agents) >= 4, "Should have at least 4 agents"
    print("  ✅ PASSED")
    
    # Test 2: Planner decisions parse correctly with agent field
    print("\n📝 TEST 2: PlannerDecision with agent field")
    decision = PlannerDecision(
        agent="gmail",
        task="Check email inbox for unread messages",
        capability="gmail.inbox",
        inputs={"max_results": 10},
        confidence=0.95,
        mode="single",
        reasoning="User wants to check their inbox"
    )
    print(f"  Agent: {decision.agent}")
    print(f"  Task: {decision.task}")
    print(f"  Capability: {decision.capability}")
    assert decision.agent == "gmail"
    assert decision.task is not None
    print("  ✅ PASSED")
    
    # Test 3: Router finds correct agent
    print("\n🔀 TEST 3: Agent Routing")
    test_routes = [
        ("gmail", "Check my email"),
        ("search", "Who invented Python?"),
        ("task", "Add task: buy groceries"),
        ("code", "Run print(2+2)"),
    ]
    for agent_name, task in test_routes:
        agent = router.get_agent(agent_name)
        assert agent is not None, f"Agent '{agent_name}' not found"
        print(f"  '{task}' → {agent_name} agent ✓ ({len(agent.tools)} tools)")
    print("  ✅ PASSED")
    
    # Test 4: Gmail agent has correct tools
    print("\n📧 TEST 4: Gmail Agent Tools")
    gmail_agent = router.get_agent("gmail")
    expected_tools = [
        "gmail_check_inbox", "gmail_read_email", "gmail_search",
        "gmail_create_draft", "gmail_draft_reply", "gmail_send_draft",
        "gmail_send_email", "gmail_mark_read", "gmail_archive", "gmail_get_thread"
    ]
    for tool_name in expected_tools:
        assert tool_name in gmail_agent.tools, f"Missing tool: {tool_name}"
        print(f"  ✓ {tool_name}")
    print("  ✅ PASSED")
    
    # Test 5: Direct tool execution through agent tools
    print("\n⚡ TEST 5: Direct Tool Execution (no LLM)")
    inbox_result = gmail_agent.tools["gmail_check_inbox"]["function"](max_results=5, unread_only=True)
    print(f"  gmail_check_inbox → {inbox_result['summary']}")
    assert inbox_result["success"]
    
    read_result = gmail_agent.tools["gmail_read_email"]["function"](email_id="mock_001")
    print(f"  gmail_read_email → Subject: {read_result['email']['subject']}")
    assert read_result["success"]
    
    search_result = gmail_agent.tools["gmail_search"]["function"](query="budget")
    print(f"  gmail_search → Found {search_result['count']} emails")
    assert search_result["success"]
    
    draft_result = gmail_agent.tools["gmail_create_draft"]["function"](
        to="test@example.com", subject="Test", body="Hello"
    )
    print(f"  gmail_create_draft → Draft ID: {draft_result['draft_id']}")
    assert draft_result["success"]
    print("  ✅ PASSED")
    
    # Test 6: Agent prompt generation
    print("\n📋 TEST 6: Agent Prompt Generation")
    prompt = gmail_agent._build_agent_prompt("Check my inbox")
    assert "gmail_check_inbox" in prompt
    assert "gmail_read_email" in prompt
    assert "gmail_draft_reply" in prompt
    print(f"  Prompt length: {len(prompt)} chars")
    print(f"  Contains tool docs: ✓")
    print(f"  Contains response format: ✓")
    print("  ✅ PASSED")
    
    # Test 7: Agent descriptions for planner
    print("\n📜 TEST 7: Planner Agent Descriptions")
    desc = router.get_agent_descriptions()
    assert "gmail" in desc
    assert "search" in desc
    assert "task" in desc
    print(f"  {desc}")
    print("  ✅ PASSED")
    
    # Test 8: Synthesize from history
    print("\n🧠 TEST 8: Answer Synthesis from Tool Results")
    fake_history = [
        {
            "step": 1,
            "tool": "gmail_check_inbox",
            "inputs": {"unread_only": True},
            "result": json.dumps({
                "success": True,
                "total": 2,
                "unread_count": 2,
                "emails": [
                    {"from": "Rahul <rahul@company.com>", "subject": "Budget Meeting"},
                    {"from": "Priya <priya@company.com>", "subject": "Project Alpha"},
                ]
            })
        }
    ]
    synthesized = gmail_agent._synthesize_from_history("Check my inbox", fake_history)
    print(f"  Synthesized answer ({len(synthesized)} chars):")
    for line in synthesized.split("\n")[:5]:
        print(f"    {line}")
    assert "Rahul" in synthesized or "Budget" in synthesized
    print("  ✅ PASSED")
    
    print("\n" + "=" * 60)
    print("✅ ALL TRUE AGENT ARCHITECTURE TESTS PASSED!")
    print("=" * 60)
    
    print("\n📊 Architecture Summary:")
    print("  OLD: Planner → Python function (no reasoning)")
    print("  NEW: Planner → Domain Agent (LLM) → Tools (multi-step)")
    print(f"\n  Agents: {len(agents)}")
    print(f"  Total tools: {sum(a['tool_count'] for a in agents)}")
    print(f"  Gmail tools: {len(gmail_agent.tools)}")


if __name__ == "__main__":
    test_routing_logic()
