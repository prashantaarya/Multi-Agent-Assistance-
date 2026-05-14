"""
Test if the Groq model is working correctly
"""
import os
import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo

load_dotenv()

# Create model client
model_client = OpenAIChatCompletionClient(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    model_info=ModelInfo(
        family="llama",
        vision=False,
        function_calling=True,
        structured_output=False,
        json_output=False,
    ),
    create_config={
        "temperature": 0.7,  # Increased from 0.3
        "max_tokens": 2000,  # Increased from 1000
        "top_p": 0.9,
    },
)

async def test_simple_prompt():
    """Test with a very simple prompt"""
    print("=" * 60)
    print("TEST 1: Simple prompt")
    print("=" * 60)
    
    agent = AssistantAgent(
        name="test_agent",
        model_client=model_client,
        system_message="You are a helpful assistant. Respond in natural language."
    )
    
    team = RoundRobinGroupChat([agent], max_turns=1)
    task = "What is 2+2?"
    
    print(f"\nTask: {task}")
    print("\nResponse:")
    
    buffer = ""
    async for msg in team.run_stream(task=task):
        if hasattr(msg, 'content') and msg.content:
            buffer += msg.content
            print(msg.content, end="", flush=True)
    
    print(f"\n\nBuffer length: {len(buffer)}")
    print(f"Is echoing? {buffer.strip() == task}")
    return buffer

async def test_json_prompt():
    """Test with JSON output requirement"""
    print("\n" + "=" * 60)
    print("TEST 2: JSON response")
    print("=" * 60)
    
    agent = AssistantAgent(
        name="test_agent",
        model_client=model_client,
        system_message="""You must respond in JSON format.

Example:
{
  "thought": "I need to search for jobs",
  "tool": "search_jobs",
  "tool_inputs": {"query": "AI Engineer"},
  "is_final": false
}

Respond ONLY with valid JSON."""
    )
    
    team = RoundRobinGroupChat([agent], max_turns=1)
    task = "Find AI Engineer jobs in Bangalore"
    
    print(f"\nTask: {task}")
    print("\nResponse:")
    
    buffer = ""
    async for msg in team.run_stream(task=task):
        if hasattr(msg, 'content') and msg.content:
            buffer += msg.content
            print(msg.content, end="", flush=True)
    
    print(f"\n\nBuffer length: {len(buffer)}")
    print(f"Is echoing? {buffer.strip() == task}")
    return buffer

async def test_with_instruction_prefix():
    """Test with clear instruction prefix"""
    print("\n" + "=" * 60)
    print("TEST 3: With instruction prefix")
    print("=" * 60)
    
    agent = AssistantAgent(
        name="test_agent",
        model_client=model_client,
        system_message="You are a helpful assistant that responds with JSON."
    )
    
    team = RoundRobinGroupChat([agent], max_turns=1)
    task = "INSTRUCTION: Generate a JSON response for this request: Find jobs\n\nYour JSON:"
    
    print(f"\nTask: {task}")
    print("\nResponse:")
    
    buffer = ""
    async for msg in team.run_stream(task=task):
        if hasattr(msg, 'content') and msg.content:
            buffer += msg.content
            print(msg.content, end="", flush=True)
    
    print(f"\n\nBuffer length: {len(buffer)}")
    return buffer

async def main():
    print("Testing Groq model behavior...\n")
    
    try:
        await test_simple_prompt()
        await test_json_prompt()
        await test_with_instruction_prefix()
        
        print("\n" + "=" * 60)
        print("CONCLUSION:")
        print("=" * 60)
        print("If all tests are echoing, the issue is with the Groq API or model.")
        print("If some work and some don't, the issue is with prompt formatting.")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
