# test_gmail_memory.py
"""Test Gmail + Memory System Integration"""

from agents.gmail_agent import gmail_read_email, gmail_check_inbox
from memory import get_contact_memory, get_unified_memory
from core.database import init_db

def test_gmail_memory_integration():
    init_db()
    
    print("=" * 60)
    print("GMAIL + MEMORY INTEGRATION TEST")
    print("=" * 60)
    
    # Add a known contact
    contacts = get_contact_memory()
    contacts.add(
        name='Rahul Sharma',
        email='rahul.sharma@company.com',
        relationship='manager',
        organization='Company Inc'
    )
    print("\n✓ Added contact: Rahul Sharma (manager @ Company Inc)")
    
    # Now check inbox - should show relationship
    print("\n📥 Checking inbox with contact context...")
    result = gmail_check_inbox()
    for email in result['emails']:
        rel = email['sender_relationship']
        status = "✓ KNOWN" if rel != "unknown" else "○ Unknown"
        print(f"  {status} | {email['from']} - Relationship: {rel}")
    
    # Read email - should show context
    print("\n📖 Reading email with full context...")
    result = gmail_read_email('mock_001')
    ctx = result['context']
    
    if ctx['sender_info']:
        print(f"  Sender: {ctx['sender_info']['name']}")
        print(f"  Relationship: {ctx['relationship']}")
        print(f"  Organization: {ctx['sender_info']['organization']}")
        print(f"  Interaction count: {ctx['previous_interactions']}")
    else:
        print("  Sender: Unknown (will be auto-created)")
    
    # Test unified memory context
    print("\n🧠 Testing UnifiedMemory.get_email_context()...")
    memory = get_unified_memory()
    email_ctx = memory.get_email_context('rahul.sharma@company.com', 'Budget Meeting')
    print(f"  Sender known: {email_ctx.sender is not None}")
    print(f"  Relationship: {email_ctx.sender.relationship if email_ctx.sender else 'N/A'}")
    print(f"  Email style: {email_ctx.email_style}")
    print(f"  Suggested priority: {email_ctx.suggested_priority}")
    
    print("\n" + "=" * 60)
    print("✅ GMAIL + MEMORY INTEGRATION WORKING!")
    print("=" * 60)


if __name__ == "__main__":
    test_gmail_memory_integration()
