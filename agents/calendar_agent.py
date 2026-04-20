# agents/calendar_agent.py
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from autogen_agentchat.agents import AssistantAgent
from core.capabilities import register
from core.schemas import ToolParameter, ParameterType

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Mock Mode Toggle (Development Mode - Active by Default)
# ──────────────────────────────────────────────────────────────────────────────
MOCK_MODE = True  # Set to False when OAuth is configured

# ──────────────────────────────────────────────────────────────────────────────
# Real Google Calendar API Setup (COMMENTED OUT - Enable after OAuth setup)
# ──────────────────────────────────────────────────────────────────────────────
# from google.oauth2.credentials import Credentials
# from google.auth.transport.requests import Request
# from google_auth_oauthlib.flow import InstalledAppFlow
# from googleapiclient.discovery import build
# import os
# import pickle
#
# SCOPES = ['https://www.googleapis.com/auth/calendar']
# TOKEN_PATH = 'token_calendar.pickle'
# CREDENTIALS_PATH = 'credentials.json'
#
# def get_calendar_service():
#     """Get authenticated Google Calendar service"""
#     creds = None
#     if os.path.exists(TOKEN_PATH):
#         with open(TOKEN_PATH, 'rb') as token:
#             creds = pickle.load(token)
#     
#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
#             creds = flow.run_local_server(port=0)
#         
#         with open(TOKEN_PATH, 'wb') as token:
#             pickle.dump(creds, token)
#     
#     return build('calendar', 'v3', credentials=creds)

# ──────────────────────────────────────────────────────────────────────────────
# Mock Data Store (Active Development Mode)
# ──────────────────────────────────────────────────────────────────────────────
MOCK_EVENTS = [
    {
        "id": "mock_event_1",
        "summary": "Team Standup",
        "description": "Daily team sync meeting",
        "start": {"dateTime": "2026-04-16T10:00:00", "timeZone": "UTC"},
        "end": {"dateTime": "2026-04-16T10:30:00", "timeZone": "UTC"},
        "attendees": [
            {"email": "alice@company.com", "responseStatus": "accepted"},
            {"email": "bob@company.com", "responseStatus": "tentative"}
        ],
        "location": "Conference Room A",
        "reminders": {"useDefault": False, "overrides": [{"method": "popup", "minutes": 10}]}
    },
    {
        "id": "mock_event_2",
        "summary": "Project Review",
        "description": "Q2 project milestone review",
        "start": {"dateTime": "2026-04-17T14:00:00", "timeZone": "UTC"},
        "end": {"dateTime": "2026-04-17T15:30:00", "timeZone": "UTC"},
        "attendees": [
            {"email": "manager@company.com", "responseStatus": "accepted"}
        ],
        "location": "Zoom Meeting",
        "reminders": {"useDefault": False, "overrides": [{"method": "email", "minutes": 30}]}
    },
    {
        "id": "mock_event_3",
        "summary": "Lunch with Client",
        "description": "Business lunch discussion",
        "start": {"dateTime": "2026-04-18T12:00:00", "timeZone": "UTC"},
        "end": {"dateTime": "2026-04-18T13:00:00", "timeZone": "UTC"},
        "attendees": [
            {"email": "client@external.com", "responseStatus": "accepted"}
        ],
        "location": "Downtown Restaurant"
    }
]


class CalendarAgent(AssistantAgent):
    """
    Google Calendar integration with LLM reasoning.
    Supports 8 capabilities for full calendar management.
    
    Development Mode: Uses mock data (OAuth code commented out)
    Production Mode: Requires OAuth 2.0 setup
    """

    def __init__(self, name: str = "calendar", model_client=None):
        super().__init__(
            name=name,
            model_client=model_client,
            system_message=(
                "You are the CalendarAgent. You manage calendar events, meetings, and schedules. "
                "You can create, list, update, delete, and search calendar events. "
                "You can check availability, manage reminders, and handle attendees. "
                "Always confirm details before creating or modifying events."
            ),
        )
        
        # Mock event counter for generating unique IDs
        self._next_event_id = 4

        # ──────────────────────────────────────────────────────────────────────
        # Register 8 Calendar Capabilities
        # ──────────────────────────────────────────────────────────────────────
        
        # 1. Create Event
        register(
            capability="calendar.create_event",
            agent_name=self.name,
            handler=self.create_event,
            description="Create a new calendar event/meeting with title, time, attendees, and location",
            parameters=[
                ToolParameter(
                    name="title",
                    type=ParameterType.STRING,
                    description="Event title/summary",
                    required=True
                ),
                ToolParameter(
                    name="start_time",
                    type=ParameterType.STRING,
                    description="Start time in ISO 8601 format (e.g., '2026-04-20T14:00:00')",
                    required=True
                ),
                ToolParameter(
                    name="end_time",
                    type=ParameterType.STRING,
                    description="End time in ISO 8601 format (e.g., '2026-04-20T15:00:00')",
                    required=True
                ),
                ToolParameter(
                    name="description",
                    type=ParameterType.STRING,
                    description="Event description/notes",
                    required=False
                ),
                ToolParameter(
                    name="location",
                    type=ParameterType.STRING,
                    description="Event location or meeting link",
                    required=False
                ),
                ToolParameter(
                    name="attendees",
                    type=ParameterType.STRING,
                    description="Comma-separated email addresses of attendees",
                    required=False
                )
            ],
            category="calendar"
        )
        
        # 2. List Events
        register(
            capability="calendar.list_events",
            agent_name=self.name,
            handler=self.list_events,
            description="List upcoming calendar events within a date range",
            parameters=[
                ToolParameter(
                    name="days_ahead",
                    type=ParameterType.STRING,
                    description="Number of days ahead to fetch (default: 7)",
                    required=False,
                    default="7"
                ),
                ToolParameter(
                    name="max_results",
                    type=ParameterType.STRING,
                    description="Maximum number of events to return (default: 10)",
                    required=False,
                    default="10"
                )
            ],
            category="calendar"
        )
        
        # 3. Update Event
        register(
            capability="calendar.update_event",
            agent_name=self.name,
            handler=self.update_event,
            description="Update an existing calendar event by ID or title",
            parameters=[
                ToolParameter(
                    name="event_identifier",
                    type=ParameterType.STRING,
                    description="Event ID or title to update",
                    required=True
                ),
                ToolParameter(
                    name="new_title",
                    type=ParameterType.STRING,
                    description="New event title (optional)",
                    required=False
                ),
                ToolParameter(
                    name="new_start_time",
                    type=ParameterType.STRING,
                    description="New start time in ISO 8601 format (optional)",
                    required=False
                ),
                ToolParameter(
                    name="new_end_time",
                    type=ParameterType.STRING,
                    description="New end time in ISO 8601 format (optional)",
                    required=False
                ),
                ToolParameter(
                    name="new_location",
                    type=ParameterType.STRING,
                    description="New location (optional)",
                    required=False
                )
            ],
            category="calendar"
        )
        
        # 4. Delete Event
        register(
            capability="calendar.delete_event",
            agent_name=self.name,
            handler=self.delete_event,
            description="Delete a calendar event by ID or title",
            parameters=[
                ToolParameter(
                    name="event_identifier",
                    type=ParameterType.STRING,
                    description="Event ID or title to delete",
                    required=True
                )
            ],
            category="calendar"
        )
        
        # 5. Search Events
        register(
            capability="calendar.search_events",
            agent_name=self.name,
            handler=self.search_events,
            description="Search calendar events by keyword in title or description",
            parameters=[
                ToolParameter(
                    name="query",
                    type=ParameterType.STRING,
                    description="Search keyword or phrase",
                    required=True
                )
            ],
            category="calendar"
        )
        
        # 6. Check Free/Busy
        register(
            capability="calendar.check_availability",
            agent_name=self.name,
            handler=self.check_availability,
            description="Check if a time slot is available (free/busy status)",
            parameters=[
                ToolParameter(
                    name="start_time",
                    type=ParameterType.STRING,
                    description="Start time in ISO 8601 format",
                    required=True
                ),
                ToolParameter(
                    name="end_time",
                    type=ParameterType.STRING,
                    description="End time in ISO 8601 format",
                    required=True
                )
            ],
            category="calendar"
        )
        
        # 7. Set Reminder
        register(
            capability="calendar.set_reminder",
            agent_name=self.name,
            handler=self.set_reminder,
            description="Add or update reminder for an event",
            parameters=[
                ToolParameter(
                    name="event_identifier",
                    type=ParameterType.STRING,
                    description="Event ID or title",
                    required=True
                ),
                ToolParameter(
                    name="minutes_before",
                    type=ParameterType.STRING,
                    description="Minutes before event to remind (e.g., '10', '30', '60')",
                    required=True
                ),
                ToolParameter(
                    name="method",
                    type=ParameterType.STRING,
                    description="Reminder method: 'popup' or 'email'",
                    required=False,
                    default="popup"
                )
            ],
            category="calendar"
        )
        
        # 8. Manage Attendees
        register(
            capability="calendar.manage_attendees",
            agent_name=self.name,
            handler=self.manage_attendees,
            description="Add or remove attendees from a calendar event",
            parameters=[
                ToolParameter(
                    name="event_identifier",
                    type=ParameterType.STRING,
                    description="Event ID or title",
                    required=True
                ),
                ToolParameter(
                    name="action",
                    type=ParameterType.STRING,
                    description="Action to perform: 'add' or 'remove'",
                    required=True
                ),
                ToolParameter(
                    name="email",
                    type=ParameterType.STRING,
                    description="Attendee email address",
                    required=True
                )
            ],
            category="calendar"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Capability Handlers (Mock Mode Active)
    # ──────────────────────────────────────────────────────────────────────────
    
    async def create_event(
        self,
        title: str,
        start_time: str,
        end_time: str,
        description: str = "",
        location: str = "",
        attendees: str = ""
    ) -> str:
        """Create a new calendar event"""
        
        if MOCK_MODE:
            # Mock data implementation
            event_id = f"mock_event_{self._next_event_id}"
            self._next_event_id += 1
            
            attendee_list = []
            if attendees:
                for email in [e.strip() for e in attendees.split(",")]:
                    attendee_list.append({"email": email, "responseStatus": "needsAction"})
            
            new_event = {
                "id": event_id,
                "summary": title,
                "description": description,
                "start": {"dateTime": start_time, "timeZone": "UTC"},
                "end": {"dateTime": end_time, "timeZone": "UTC"},
                "location": location,
                "attendees": attendee_list
            }
            
            MOCK_EVENTS.append(new_event)
            
            response_text = f"""✅ Event created successfully!

📅 **{title}**
🕒 Start: {start_time}
🕒 End: {end_time}
📍 Location: {location or 'Not specified'}
👥 Attendees: {len(attendee_list)} person(s)
🆔 Event ID: {event_id}

[MOCK MODE - No actual calendar event created]"""
            
            return {
                "response": response_text,
                "data": {
                    "event": {
                        "id": event_id,
                        "title": title,
                        "description": description,
                        "start": start_time,
                        "end": end_time,
                        "location": location,
                        "attendees": [e.strip() for e in attendees.split(",") if e.strip()] if attendees else [],
                        "created": True
                    }
                }
            }
        
        # Real implementation (commented out)
        # else:
        #     service = get_calendar_service()
        #     event = {
        #         'summary': title,
        #         'description': description,
        #         'location': location,
        #         'start': {'dateTime': start_time, 'timeZone': 'UTC'},
        #         'end': {'dateTime': end_time, 'timeZone': 'UTC'},
        #     }
        #     
        #     if attendees:
        #         event['attendees'] = [{'email': e.strip()} for e in attendees.split(',')]
        #     
        #     result = service.events().insert(calendarId='primary', body=event).execute()
        #     return f"✅ Event created: {result.get('htmlLink')}"

    async def list_events(self, days_ahead: str = "7", max_results: str = "10"):
        """List upcoming calendar events - returns structured data"""
        
        if MOCK_MODE:
            days = int(days_ahead)
            max_res = int(max_results)
            
            # Filter events within the date range
            now = datetime.now()
            future_date = now + timedelta(days=days)
            
            upcoming = []
            for event in MOCK_EVENTS:
                event_start = datetime.fromisoformat(event["start"]["dateTime"].replace("Z", ""))
                if now <= event_start <= future_date:
                    upcoming.append(event)
            
            upcoming = upcoming[:max_res]
            
            if not upcoming:
                return {
                    "response": f"📅 No events found in the next {days} days.",
                    "data": {
                        "events": [],
                        "total": 0,
                        "date_range": {"start": now.isoformat(), "end": future_date.isoformat()},
                        "days": days
                    }
                }
            
            # Build human-readable response
            result = f"📅 **Upcoming Events** (next {days} days):\n\n"
            for i, event in enumerate(upcoming, 1):
                result += f"{i}. **{event['summary']}**\n"
                result += f"   🕒 {event['start']['dateTime']}\n"
                if event.get('location'):
                    result += f"   📍 {event['location']}\n"
                if event.get('attendees'):
                    result += f"   👥 {len(event['attendees'])} attendee(s)\n"
                result += "\n"
            
            result += "[MOCK MODE - Showing mock calendar data]"
            
            # Build structured data
            events_data = []
            for event in upcoming:
                events_data.append({
                    "id": event["id"],
                    "title": event["summary"],
                    "description": event.get("description", ""),
                    "start": event["start"]["dateTime"],
                    "end": event["end"]["dateTime"],
                    "location": event.get("location", ""),
                    "attendees": [a["email"] for a in event.get("attendees", [])]
                })
            
            return {
                "response": result,
                "data": {
                    "events": events_data,
                    "total": len(upcoming),
                    "date_range": {"start": now.isoformat(), "end": future_date.isoformat()},
                    "days": days
                }
            }
        
        # Real implementation (commented out)
        # else:
        #     service = get_calendar_service()
        #     now = datetime.utcnow().isoformat() + 'Z'
        #     
        #     events_result = service.events().list(
        #         calendarId='primary',
        #         timeMin=now,
        #         maxResults=int(max_results),
        #         singleEvents=True,
        #         orderBy='startTime'
        #     ).execute()
        #     
        #     events = events_result.get('items', [])
        #     # Format and return events...

    async def update_event(
        self,
        event_identifier: str,
        new_title: str = "",
        new_start_time: str = "",
        new_end_time: str = "",
        new_location: str = ""
    ) -> str:
        """Update an existing calendar event"""
        
        if MOCK_MODE:
            # Find event by ID or title
            event = None
            for e in MOCK_EVENTS:
                if e["id"] == event_identifier or e["summary"].lower() == event_identifier.lower():
                    event = e
                    break
            
            if not event:
                return f"❌ Event not found: {event_identifier}"
            
            # Update fields
            if new_title:
                event["summary"] = new_title
            if new_start_time:
                event["start"]["dateTime"] = new_start_time
            if new_end_time:
                event["end"]["dateTime"] = new_end_time
            if new_location:
                event["location"] = new_location
            
            return f"""✅ Event updated successfully!

📅 **{event['summary']}**
🕒 Start: {event['start']['dateTime']}
🕒 End: {event['end']['dateTime']}
📍 Location: {event.get('location', 'Not specified')}

[MOCK MODE - No actual calendar event updated]"""
        
        # Real implementation (commented out)
        # else:
        #     service = get_calendar_service()
        #     event = service.events().get(calendarId='primary', eventId=event_identifier).execute()
        #     # Update and patch event...

    async def delete_event(self, event_identifier: str) -> str:
        """Delete a calendar event"""
        
        if MOCK_MODE:
            global MOCK_EVENTS
            original_count = len(MOCK_EVENTS)
            
            MOCK_EVENTS = [
                e for e in MOCK_EVENTS
                if e["id"] != event_identifier and e["summary"].lower() != event_identifier.lower()
            ]
            
            if len(MOCK_EVENTS) < original_count:
                return f"✅ Event deleted: {event_identifier}\n\n[MOCK MODE]"
            else:
                return f"❌ Event not found: {event_identifier}"
        
        # Real implementation (commented out)
        # else:
        #     service = get_calendar_service()
        #     service.events().delete(calendarId='primary', eventId=event_identifier).execute()
        #     return f"✅ Event deleted: {event_identifier}"

    async def search_events(self, query: str):
        """Search calendar events by keyword - returns structured data"""
        
        if MOCK_MODE:
            query_lower = query.lower()
            matches = [
                e for e in MOCK_EVENTS
                if query_lower in e["summary"].lower() or query_lower in e.get("description", "").lower()
            ]
            
            if not matches:
                return {
                    "response": f"🔍 No events found matching: {query}",
                    "data": {"events": [], "query": query, "total": 0}
                }
            
            result = f"🔍 **Search Results** for '{query}':\n\n"
            for i, event in enumerate(matches, 1):
                result += f"{i}. **{event['summary']}**\n"
                result += f"   🕒 {event['start']['dateTime']}\n"
                if event.get('location'):
                    result += f"   📍 {event['location']}\n"
                result += "\n"
            
            result += "[MOCK MODE]"
            
            events_data = [
                {
                    "id": e["id"],
                    "title": e["summary"],
                    "start": e["start"]["dateTime"],
                    "end": e["end"]["dateTime"],
                    "location": e.get("location", "")
                }
                for e in matches
            ]
            
            return {
                "response": result,
                "data": {"events": events_data, "query": query, "total": len(matches)}
            }
        
        # Real implementation (commented out)
        # else:
        #     service = get_calendar_service()
        #     events_result = service.events().list(
        #         calendarId='primary',
        #         q=query,
        #         singleEvents=True,
        #         orderBy='startTime'
        #     ).execute()
        #     # Format and return results...

    async def check_availability(self, start_time: str, end_time: str):
        """Check if a time slot is available - returns structured data"""
        
        if MOCK_MODE:
            start_dt = datetime.fromisoformat(start_time.replace("Z", ""))
            end_dt = datetime.fromisoformat(end_time.replace("Z", ""))
            
            conflicts = []
            for event in MOCK_EVENTS:
                event_start = datetime.fromisoformat(event["start"]["dateTime"].replace("Z", ""))
                event_end = datetime.fromisoformat(event["end"]["dateTime"].replace("Z", ""))
                
                # Check overlap
                if not (end_dt <= event_start or start_dt >= event_end):
                    conflicts.append(event)
            
            if not conflicts:
                return {
                    "response": f"✅ Time slot is **AVAILABLE**\n🕒 {start_time} to {end_time}\n\n[MOCK MODE]",
                    "data": {
                        "available": True,
                        "start": start_time,
                        "end": end_time,
                        "conflicts": []
                    }
                }
            else:
                result = f"⚠️ Time slot has **CONFLICTS**\n🕒 {start_time} to {end_time}\n\n"
                result += "**Conflicting events:**\n"
                for event in conflicts:
                    result += f"- {event['summary']} ({event['start']['dateTime']})\n"
                result += "\n[MOCK MODE]"
                
                conflicts_data = [
                    {
                        "id": e["id"],
                        "title": e["summary"],
                        "start": e["start"]["dateTime"],
                        "end": e["end"]["dateTime"]
                    }
                    for e in conflicts
                ]
                
                return {
                    "response": result,
                    "data": {
                        "available": False,
                        "start": start_time,
                        "end": end_time,
                        "conflicts": conflicts_data
                    }
                }
        
        # Real implementation (commented out)
        # else:
        #     service = get_calendar_service()
        #     body = {
        #         'timeMin': start_time,
        #         'timeMax': end_time,
        #         'items': [{'id': 'primary'}]
        #     }
        #     result = service.freebusy().query(body=body).execute()
        #     # Check busy periods...

    async def set_reminder(
        self,
        event_identifier: str,
        minutes_before: str,
        method: str = "popup"
    ) -> str:
        """Set reminder for an event"""
        
        if MOCK_MODE:
            event = None
            for e in MOCK_EVENTS:
                if e["id"] == event_identifier or e["summary"].lower() == event_identifier.lower():
                    event = e
                    break
            
            if not event:
                return f"❌ Event not found: {event_identifier}"
            
            reminder = {
                "method": method,
                "minutes": int(minutes_before)
            }
            
            if "reminders" not in event:
                event["reminders"] = {"useDefault": False, "overrides": []}
            
            event["reminders"]["overrides"].append(reminder)
            
            return f"""✅ Reminder set for **{event['summary']}**
⏰ {minutes_before} minutes before
📬 Method: {method}

[MOCK MODE]"""
        
        # Real implementation (commented out)
        # else:
        #     service = get_calendar_service()
        #     event = service.events().get(calendarId='primary', eventId=event_identifier).execute()
        #     # Update reminders and patch...

    async def manage_attendees(self, event_identifier: str, action: str, email: str) -> str:
        """Add or remove attendees from an event"""
        
        if MOCK_MODE:
            event = None
            for e in MOCK_EVENTS:
                if e["id"] == event_identifier or e["summary"].lower() == event_identifier.lower():
                    event = e
                    break
            
            if not event:
                return f"❌ Event not found: {event_identifier}"
            
            if "attendees" not in event:
                event["attendees"] = []
            
            if action.lower() == "add":
                # Check if already exists
                if any(a["email"] == email for a in event["attendees"]):
                    return f"ℹ️ {email} is already an attendee"
                
                event["attendees"].append({"email": email, "responseStatus": "needsAction"})
                return f"""✅ Attendee added to **{event['summary']}**
👤 {email}
📧 Invitation sent (mock)

[MOCK MODE]"""
            
            elif action.lower() == "remove":
                original_count = len(event["attendees"])
                event["attendees"] = [a for a in event["attendees"] if a["email"] != email]
                
                if len(event["attendees"]) < original_count:
                    return f"✅ Attendee removed from **{event['summary']}**: {email}\n\n[MOCK MODE]"
                else:
                    return f"❌ Attendee not found: {email}"
            
            else:
                return f"❌ Invalid action: {action}. Use 'add' or 'remove'"
        
        # Real implementation (commented out)
        # else:
        #     service = get_calendar_service()
        #     event = service.events().get(calendarId='primary', eventId=event_identifier).execute()
        #     # Modify attendees and patch...
