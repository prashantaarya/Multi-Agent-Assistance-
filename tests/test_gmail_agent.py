# tests/test_gmail_agent.py
"""
Gmail Agent Test Suite (Mock Mode)
Run: python -m pytest tests/test_gmail_agent.py -v
Or:  python tests/test_gmail_agent.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.gmail_agent import (
    gmail_check_inbox,
    gmail_read_email,
    gmail_search,
    gmail_create_draft,
    gmail_draft_reply,
    gmail_send_draft,
    gmail_send_email,
    gmail_mark_read,
    gmail_archive,
    gmail_get_thread,
)
from core.database import init_db

def test_gmail_agent():
    # Initialize database
    init_db()
    
    print("=" * 60)
    print("GMAIL AGENT TEST (Mock Mode)")
    print("=" * 60)
    
    # Test 1: Check inbox
    print("\n📥 TEST 1: Check Inbox")
    result = gmail_check_inbox(max_results=5, unread_only=True)
    print(f"  Status: {result['success']}")
    print(f"  Summary: {result['summary']}")
    for email in result['emails']:
        print(f"  - From: {email['from']}")
        print(f"    Subject: {email['subject']}")
        print(f"    Priority: {email['suggested_priority']} | Relationship: {email['sender_relationship']}")
    assert result['success'], "Inbox check failed"
    
    # Test 2: Read specific email
    print("\n📖 TEST 2: Read Email")
    result = gmail_read_email('mock_001')
    print(f"  Subject: {result['email']['subject']}")
    print(f"  From: {result['email']['from']}")
    print(f"  Body:\n    {result['email']['body'][:100]}...")
    print(f"  Context - Relationship: {result['context']['relationship']}")
    assert result['success'], "Read email failed"
    
    # Test 3: Search emails
    print("\n🔍 TEST 3: Search Emails")
    result = gmail_search('budget')
    print(f"  Found {result['count']} emails matching 'budget'")
    for email in result['emails']:
        print(f"  - {email['subject']} from {email['from']}")
    assert result['success'], "Search failed"
    
    # Test 4: Create draft
    print("\n✏️ TEST 4: Create Draft")
    result = gmail_create_draft(
        to='rahul.sharma@company.com',
        subject='Re: Q1 Budget Review Meeting',
        body='Hi Rahul,\n\nSure, I am available tomorrow at 2 PM.\n\nBest regards'
    )
    print(f"  Draft ID: {result['draft_id']}")
    print(f"  To: {result['preview']['to']}")
    print(f"  Subject: {result['preview']['subject']}")
    assert result['success'], "Create draft failed"
    draft_id = result['draft_id']
    
    # Test 5: Draft reply
    print("\n↩️ TEST 5: Draft Reply")
    result = gmail_draft_reply(
        email_id='mock_002',
        body='Thanks for the update, Priya! Great progress on Project Alpha.'
    )
    print(f"  Reply Draft ID: {result['draft_id']}")
    print(f"  Replying to: {result['replying_to']['from']}")
    assert result['success'], "Draft reply failed"
    reply_draft_id = result['draft_id']
    
    # Test 6: Send draft (mock)
    print("\n📤 TEST 6: Send Draft (Mock)")
    result = gmail_send_draft(reply_draft_id)
    print(f"  Sent: {result['success']} - {result['message']}")
    assert result['success'], "Send draft failed"
    
    # Test 7: Send email directly (mock)
    print("\n📧 TEST 7: Send Email Directly (Mock)")
    result = gmail_send_email(
        to='team@company.com',
        subject='Quick Update',
        body='FYI - everything is on track.'
    )
    print(f"  Sent: {result['success']} - {result['message']}")
    assert result['success'], "Send email failed"
    
    # Test 8: Get thread
    print("\n💬 TEST 8: Get Thread")
    result = gmail_get_thread('thread_001')
    print(f"  Thread has {result['message_count']} messages")
    assert result['success'], "Get thread failed"
    
    # Test 9: Mark as read
    print("\n✅ TEST 9: Mark Read")
    result = gmail_mark_read('mock_001')
    print(f"  Marked as read: {result['success']}")
    assert result['success'], "Mark read failed"
    
    # Test 10: Archive
    print("\n📦 TEST 10: Archive")
    result = gmail_archive('mock_003')
    print(f"  Archived: {result['success']}")
    assert result['success'], "Archive failed"
    
    print("\n" + "=" * 60)
    print("✅ ALL GMAIL TESTS PASSED!")
    print("=" * 60)
    
    # Summary
    print("\nGmail Agent Capabilities Verified:")
    print("  ✓ gmail_check_inbox - Check for unread emails")
    print("  ✓ gmail_read_email - Read full email content")
    print("  ✓ gmail_search - Search with query syntax")
    print("  ✓ gmail_create_draft - Create new drafts")
    print("  ✓ gmail_draft_reply - Create reply drafts")
    print("  ✓ gmail_send_draft - Send existing drafts")
    print("  ✓ gmail_send_email - Send emails directly")
    print("  ✓ gmail_get_thread - Get conversation threads")
    print("  ✓ gmail_mark_read - Mark emails as read")
    print("  ✓ gmail_archive - Archive emails")


if __name__ == "__main__":
    test_gmail_agent()
