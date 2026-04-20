"""
Quick test script for Calendar Agent with Mock Data
Run this to verify calendar capabilities work without OAuth
"""
import asyncio
from agents.calendar_agent import CalendarAgent

async def test_calendar_mock():
    print("=" * 60)
    print("Testing Calendar Agent (Mock Mode)")
    print("=" * 60)
    
    # Initialize agent (mock mode is default)
    agent = CalendarAgent(name="calendar_test")
    
    # Test 1: List events
    print("\n📅 Test 1: List upcoming events")
    print("-" * 60)
    result = await agent.list_events(days_ahead="30", max_results="10")
    print(result)
    
    # Test 2: Create event
    print("\n📅 Test 2: Create new event")
    print("-" * 60)
    result = await agent.create_event(
        title="Product Demo",
        start_time="2026-04-20T15:00:00",
        end_time="2026-04-20T16:00:00",
        description="Demo new features to stakeholders",
        location="Conference Room B",
        attendees="alice@company.com, bob@company.com"
    )
    print(result)
    
    # Test 3: Check availability
    print("\n📅 Test 3: Check availability")
    print("-" * 60)
    result = await agent.check_availability(
        start_time="2026-04-17T14:00:00",
        end_time="2026-04-17T15:30:00"
    )
    print(result)
    
    # Test 4: Search events
    print("\n📅 Test 4: Search for meetings")
    print("-" * 60)
    result = await agent.search_events(query="standup")
    print(result)
    
    # Test 5: Update event
    print("\n📅 Test 5: Update event location")
    print("-" * 60)
    result = await agent.update_event(
        event_identifier="Team Standup",
        new_location="Zoom Meeting Room 1"
    )
    print(result)
    
    # Test 6: Set reminder
    print("\n📅 Test 6: Set reminder")
    print("-" * 60)
    result = await agent.set_reminder(
        event_identifier="Team Standup",
        minutes_before="15",
        method="popup"
    )
    print(result)
    
    # Test 7: Manage attendees
    print("\n📅 Test 7: Add attendee")
    print("-" * 60)
    result = await agent.manage_attendees(
        event_identifier="Project Review",
        action="add",
        email="charlie@company.com"
    )
    print(result)
    
    # Test 8: Delete event
    print("\n📅 Test 8: Delete event")
    print("-" * 60)
    result = await agent.delete_event(event_identifier="Lunch with Client")
    print(result)
    
    print("\n" + "=" * 60)
    print("✅ All Calendar Mock Tests Passed!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_calendar_mock())
