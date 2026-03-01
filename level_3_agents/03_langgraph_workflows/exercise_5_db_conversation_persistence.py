"""
Exercise 5: Database-Backed Conversation Persistence

Skills practiced:
- Storing multi-turn conversations in SQLite (and optionally MongoDB)
- Resuming conversations across sessions using session IDs
- Understanding how production AI systems (ChatGPT, Copilot) persist chats
- Building a conversation store with search and session management

Key insight: When you chat with ChatGPT or Copilot, close the browser,
  and come back hours later — your full conversation is still there.
  That's because every message is stored in a database keyed by a
  session/thread ID. This exercise teaches you to build that exact
  pattern from scratch.

  Exercise 4 taught persistence with JSON files. This exercise
  upgrades to a real database, which is what production systems use.

Architecture (how production AI systems do it):
  ┌─────────────┐
  │   User msg   │
  └──────┬───────┘
         │
         ▼
  ┌──────────────────────────┐
  │  Load conversation from  │◀── session_id
  │  database (all previous  │
  │  messages for this chat) │
  └──────┬───────────────────┘
         │
         ▼
  ┌──────────────────────────┐
  │  Send full history +     │
  │  new message to LLM      │──▶ LLM response
  └──────┬───────────────────┘
         │
         ▼
  ┌──────────────────────────┐
  │  Save BOTH user msg and  │
  │  assistant response to   │──▶ Database
  │  database                │
  └──────────────────────────┘

Database schema:
  sessions:   id, title, system_prompt, model, created_at, updated_at
  messages:   id, session_id, role, content, token_count, created_at
"""

import os
import json
import uuid
import sqlite3
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

client = OpenAI()

# ============================================================
# PART 1: SQLite Conversation Store
# ============================================================
# Why SQLite? It's built into Python — zero installation needed.
# The SAME pattern works with MongoDB, PostgreSQL, Redis, etc.
# We'll show the MongoDB equivalent at the end.
# ============================================================

DB_PATH = os.path.join(os.path.dirname(__file__), "conversations.db")


class ConversationStore:
    """
    Database-backed conversation store.

    This is how production AI systems persist chats:
    - Each conversation has a unique session_id (like a thread_id)
    - Every message (user + assistant) is saved with timestamps
    - Conversations can be loaded, resumed, searched, and deleted

    Think of this as building your own mini version of what
    ChatGPT uses to remember your conversations.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Sessions table — one row per conversation
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id   TEXT PRIMARY KEY,
                title        TEXT,
                system_prompt TEXT,
                model        TEXT DEFAULT 'gpt-4o-mini',
                total_turns  INTEGER DEFAULT 0,
                created_at   TEXT,
                updated_at   TEXT
            )
        """)

        # Messages table — every single message in every conversation
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,          -- 'system', 'user', or 'assistant'
                content     TEXT NOT NULL,
                token_count INTEGER DEFAULT 0,
                created_at  TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)

        # Index for fast lookups by session
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages(session_id, created_at)
        """)

        conn.commit()
        conn.close()

    # ----------------------------------------------------------
    # Session Management
    # ----------------------------------------------------------

    def create_session(
        self,
        system_prompt: str = "You are a helpful clinical assistant.",
        title: str = None,
        model: str = "gpt-4o-mini",
    ) -> str:
        """
        Create a new conversation session.
        Returns a unique session_id (like ChatGPT's thread ID).
        """
        session_id = str(uuid.uuid4())[:8]  # Short ID for readability
        now = datetime.now().isoformat()

        if title is None:
            title = f"Chat {now[:10]}"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (session_id, title, system_prompt, model, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, title, system_prompt, model, now, now),
        )

        # Also store the system prompt as the first message
        cursor.execute(
            "INSERT INTO messages (session_id, role, content, created_at) "
            "VALUES (?, 'system', ?, ?)",
            (session_id, system_prompt, now),
        )

        conn.commit()
        conn.close()
        return session_id

    def list_sessions(self) -> list[dict]:
        """List all saved conversations (like ChatGPT's sidebar)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT session_id, title, model, total_turns, created_at, updated_at "
            "FROM sessions ORDER BY updated_at DESC"
        )
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

    def get_session_info(self, session_id: str) -> Optional[dict]:
        """Get metadata about a specific session."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def delete_session(self, session_id: str):
        """Delete a conversation and all its messages."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()

    def delete_all_sessions(self):
        """Clear everything — useful for testing."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages")
        cursor.execute("DELETE FROM sessions")
        conn.commit()
        conn.close()

    # ----------------------------------------------------------
    # Message Storage & Retrieval
    # ----------------------------------------------------------

    def get_messages(self, session_id: str) -> list[dict]:
        """
        Load the full conversation history for a session.
        Returns messages in the format OpenAI expects:
          [{"role": "system", "content": "..."}, {"role": "user", ...}, ...]
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content FROM messages "
            "WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        )
        messages = [{"role": r["role"], "content": r["content"]} for r in cursor.fetchall()]
        conn.close()
        return messages

    def add_message(self, session_id: str, role: str, content: str, token_count: int = 0):
        """Save a single message to the database."""
        now = datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO messages (session_id, role, content, token_count, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, token_count, now),
        )

        # Update session metadata
        cursor.execute(
            "UPDATE sessions SET updated_at = ?, total_turns = total_turns + 1 WHERE session_id = ?",
            (now, session_id),
        )

        conn.commit()
        conn.close()

    def get_message_count(self, session_id: str) -> int:
        """How many messages in this conversation?"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?",
            (session_id,),
        )
        count = cursor.fetchone()[0]
        conn.close()
        return count

    # ----------------------------------------------------------
    # Search Across Conversations
    # ----------------------------------------------------------

    def search_messages(self, query: str, limit: int = 10) -> list[dict]:
        """
        Search across all conversations for a keyword.
        Useful for: 'Find all chats where we discussed diabetes'
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT m.session_id, s.title, m.role, m.content, m.created_at "
            "FROM messages m JOIN sessions s ON m.session_id = s.session_id "
            "WHERE m.content LIKE ? AND m.role != 'system' "
            "ORDER BY m.created_at DESC LIMIT ?",
            (f"%{query}%", limit),
        )
        results = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return results


# ============================================================
# PART 2: Persistent Chat — AI That Remembers Across Sessions
# ============================================================
# This is the core pattern: load history → send to LLM → save response
# ============================================================

class PersistentChatAgent:
    """
    A chat agent that persists every conversation to a database.

    How it works (same as ChatGPT / Copilot):
    1. User sends a message
    2. Agent loads ALL previous messages from DB for this session
    3. Sends the full history + new message to the LLM
    4. Saves the user message AND the LLM response to DB
    5. Returns the response

    Result: Close the program, restart hours later, continue
    the EXACT same conversation with full context.
    """

    def __init__(self, store: ConversationStore = None):
        self.store = store or ConversationStore()

    def start_session(
        self,
        system_prompt: str = "You are a helpful clinical assistant.",
        title: str = None,
        model: str = "gpt-4o-mini",
    ) -> str:
        """Start a new conversation and return its session_id."""
        session_id = self.store.create_session(system_prompt, title, model)
        print(f"  [New session: {session_id}]")
        return session_id

    def chat(self, session_id: str, user_message: str) -> str:
        """
        Send a message and get a response — with full persistence.

        This is the CORE LOOP of every persistent chatbot:
          1. Load history from DB
          2. Append user message
          3. Send to LLM
          4. Save user + assistant messages to DB
          5. Return response
        """
        # Step 1: Load conversation history from database
        messages = self.store.get_messages(session_id)

        if not messages:
            raise ValueError(f"Session {session_id} not found!")

        # Step 2: Add the new user message
        messages.append({"role": "user", "content": user_message})

        # Step 3: Send entire history to LLM
        session_info = self.store.get_session_info(session_id)
        model = session_info.get("model", "gpt-4o-mini") if session_info else "gpt-4o-mini"

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )

        assistant_message = response.choices[0].message.content
        usage = response.usage

        # Step 4: Save BOTH messages to database
        self.store.add_message(
            session_id, "user", user_message,
            token_count=usage.prompt_tokens if usage else 0,
        )
        self.store.add_message(
            session_id, "assistant", assistant_message,
            token_count=usage.completion_tokens if usage else 0,
        )

        return assistant_message

    def resume_session(self, session_id: str) -> list[dict]:
        """
        Resume an existing conversation — load all previous messages.
        This is what happens when you click on an old chat in ChatGPT.
        """
        messages = self.store.get_messages(session_id)
        if not messages:
            print(f"  Session {session_id} not found!")
            return []

        info = self.store.get_session_info(session_id)
        print(f"\n  Resuming session: {info['title']} ({session_id})")
        print(f"  Created: {info['created_at'][:19]}")
        print(f"  Messages: {len(messages)}")
        print(f"  Model: {info['model']}")

        # Show conversation so far
        for msg in messages:
            if msg["role"] == "system":
                continue
            role_label = "You" if msg["role"] == "user" else "AI"
            preview = msg["content"][:120].replace("\n", " ")
            print(f"  [{role_label}]: {preview}{'...' if len(msg['content']) > 120 else ''}")

        return messages


# ============================================================
# DEMO 1: Basic Persistence — Save and Resume a Conversation
# ============================================================

def demo_basic_persistence():
    """Shows the core pattern: chat, close, resume later."""
    print("\n" + "=" * 70)
    print("  DEMO 1: BASIC PERSISTENCE — SAVE & RESUME CONVERSATIONS")
    print("=" * 70)
    print("""
  This demo shows the core persistence pattern:
  1. Start a conversation → messages saved to SQLite
  2. Simulate 'closing' the app
  3. Resume the SAME conversation — full context preserved
  """)

    agent = PersistentChatAgent()

    # --- Phase 1: Initial conversation ---
    print("\n  --- Phase 1: Starting a new clinical conversation ---")
    session_id = agent.start_session(
        system_prompt="You are a clinical assistant. Keep responses concise (2-3 sentences).",
        title="Diabetes Management Discussion",
    )

    msg1 = "Patient is a 55-year-old male with Type 2 diabetes, current HbA1c is 8.2%. Currently on metformin 1000mg BID."
    print(f"\n  You: {msg1}")
    response1 = agent.chat(session_id, msg1)
    print(f"  AI: {response1}")

    msg2 = "What medication adjustments would you suggest given the suboptimal HbA1c?"
    print(f"\n  You: {msg2}")
    response2 = agent.chat(session_id, msg2)
    print(f"  AI: {response2}")

    print(f"\n  [Messages saved to database. Session ID: {session_id}]")
    print("  [Simulating app close... 💤]")

    # --- Phase 2: Resume later (simulating a new app session) ---
    print("\n\n  --- Phase 2: Resuming hours later (new app instance) ---")

    # Create a BRAND NEW agent instance (simulates restarting the app)
    new_agent = PersistentChatAgent()

    # Resume the conversation — loads all history from DB
    new_agent.resume_session(session_id)

    # Continue chatting — the LLM has full context because we loaded history
    msg3 = "The patient also has mild kidney impairment (eGFR 55). Does that change your recommendation?"
    print(f"\n  You: {msg3}")
    response3 = new_agent.chat(session_id, msg3)
    print(f"  AI: {response3}")

    print(f"\n  ✓ The AI remembered the ENTIRE conversation from Phase 1!")
    print(f"  ✓ It knows about the diabetic patient, HbA1c 8.2%, metformin — all from the DB.")

    # Show what's in the database
    print("\n  --- What's stored in the database ---")
    messages = agent.store.get_messages(session_id)
    for i, msg in enumerate(messages):
        preview = msg["content"][:80].replace("\n", " ")
        print(f"    [{i}] {msg['role']:>10}: {preview}{'...' if len(msg['content']) > 80 else ''}")

    # Cleanup
    agent.store.delete_session(session_id)
    print(f"\n  [Session cleaned up]")


# ============================================================
# DEMO 2: Multiple Conversations — Like ChatGPT's Sidebar
# ============================================================

def demo_multiple_sessions():
    """Shows managing multiple independent conversations."""
    print("\n" + "=" * 70)
    print("  DEMO 2: MULTIPLE CONVERSATIONS — LIKE CHATGPT'S SIDEBAR")
    print("=" * 70)
    print("""
  Just like ChatGPT lets you have many separate conversations,
  each with its own context, this demo creates multiple sessions.
  """)

    agent = PersistentChatAgent()

    # Create 3 different conversations
    sessions = []

    # Session 1: Cardiology
    s1 = agent.start_session(
        system_prompt="You are a cardiology specialist. Keep responses to 2 sentences.",
        title="Cardiology Consult",
    )
    agent.chat(s1, "Patient has new-onset atrial fibrillation with rapid ventricular response, HR 142.")
    sessions.append(s1)

    # Session 2: Pediatrics
    s2 = agent.start_session(
        system_prompt="You are a pediatric specialist. Keep responses to 2 sentences.",
        title="Pediatric Follow-up",
    )
    agent.chat(s2, "6-year-old with recurring ear infections, 4 episodes in the past year.")
    sessions.append(s2)

    # Session 3: Mental Health
    s3 = agent.start_session(
        system_prompt="You are a mental health counselor. Keep responses to 2 sentences.",
        title="Mental Health Screening",
    )
    agent.chat(s3, "Patient reports persistent low mood and poor sleep for 3 weeks after job loss.")
    sessions.append(s3)

    # List all sessions (like ChatGPT's sidebar)
    print("\n  📋 All Saved Conversations:")
    print("  " + "-" * 50)
    all_sessions = agent.store.list_sessions()
    for s in all_sessions:
        print(f"    [{s['session_id']}] {s['title']}")
        print(f"      Turns: {s['total_turns']}  |  Last active: {s['updated_at'][:19]}")

    # Resume one specific conversation
    print(f"\n  Resuming cardiology consult ({s1})...")
    agent.resume_session(s1)
    response = agent.chat(s1, "Would you recommend anticoagulation given a CHA2DS2-VASc score of 3?")
    print(f"\n  AI: {response}")

    # Cleanup
    for s in sessions:
        agent.store.delete_session(s)
    print(f"\n  [All sessions cleaned up]")


# ============================================================
# DEMO 3: Search Across Conversations
# ============================================================

def demo_search_conversations():
    """Shows searching across all stored conversations."""
    print("\n" + "=" * 70)
    print("  DEMO 3: SEARCH ACROSS ALL CONVERSATIONS")
    print("=" * 70)
    print("""
  Production systems need to search across conversations:
  'Find all chats where we discussed warfarin'
  'Show me recent conversations about diabetes'
  """)

    agent = PersistentChatAgent()

    # Create conversations with different topics
    s1 = agent.start_session(title="Diabetes Case", system_prompt="Be concise (1-2 sentences).")
    agent.chat(s1, "Patient on metformin and glipizide with HbA1c 9.1%. Consider adding insulin?")

    s2 = agent.start_session(title="Anticoagulation Case", system_prompt="Be concise (1-2 sentences).")
    agent.chat(s2, "Patient on warfarin with INR of 4.2. How should we adjust the dose?")

    s3 = agent.start_session(title="Diabetes + Heart Disease", system_prompt="Be concise (1-2 sentences).")
    agent.chat(s3, "Diabetic patient with new CHF diagnosis. Should we switch from metformin to something else?")

    # Search for 'diabetes' across all conversations
    print("\n  🔍 Searching for 'diabetes' across all conversations...")
    results = agent.store.search_messages("diabet")  # Partial match
    for r in results:
        preview = r["content"][:100].replace("\n", " ")
        print(f"    [{r['session_id']}] {r['title']} | {r['role']}: {preview}...")

    # Search for 'warfarin'
    print("\n  🔍 Searching for 'warfarin'...")
    results = agent.store.search_messages("warfarin")
    for r in results:
        preview = r["content"][:100].replace("\n", " ")
        print(f"    [{r['session_id']}] {r['title']} | {r['role']}: {preview}...")

    # Cleanup
    for s in [s1, s2, s3]:
        agent.store.delete_session(s)
    print(f"\n  [All sessions cleaned up]")


# ============================================================
# DEMO 4: Interactive Persistent Chat
# ============================================================

def demo_interactive_chat():
    """Full interactive chat with persistence — try closing and resuming!"""
    print("\n" + "=" * 70)
    print("  DEMO 4: INTERACTIVE PERSISTENT CHAT")
    print("=" * 70)
    print("""
  Commands:
    'new'       — Start a new conversation
    'list'      — Show all saved conversations
    'resume'    — Resume a previous conversation
    'search X'  — Search all conversations for X
    'delete'    — Delete a conversation
    'clear'     — Delete all conversations
    'quit'      — Exit (conversations are SAVED!)

  Try this flow:
    1. Type 'new' to start a conversation
    2. Chat about a patient case
    3. Type 'quit' to exit
    4. Run this demo again
    5. Type 'list' to see your saved conversation!
    6. Type 'resume' to pick up where you left off!
  """)

    store = ConversationStore()
    agent = PersistentChatAgent(store)
    current_session = None

    while True:
        if current_session:
            user_input = input("\n  You: ").strip()
        else:
            user_input = input("\n  Command: ").strip()

        if not user_input:
            continue

        # --- Commands ---
        if user_input.lower() in ['quit', 'exit', 'q']:
            if current_session:
                info = store.get_session_info(current_session)
                count = store.get_message_count(current_session)
                print(f"\n  Session saved! ID: {current_session}")
                print(f"  Messages stored: {count}")
                print(f"  Run this demo again and type 'resume' to continue!")
            break

        elif user_input.lower() == 'new':
            title = input("  Session title (or Enter for default): ").strip()
            current_session = agent.start_session(
                system_prompt="You are a helpful clinical assistant. Be concise but thorough.",
                title=title if title else None,
            )
            print(f"  Started new session: {current_session}")
            print(f"  Start chatting! (type 'quit' to save and exit)\n")
            continue

        elif user_input.lower() == 'list':
            sessions = store.list_sessions()
            if not sessions:
                print("  No saved conversations.")
            else:
                print(f"\n  📋 Saved Conversations ({len(sessions)}):")
                for s in sessions:
                    print(f"    [{s['session_id']}] {s['title']}  ({s['total_turns']} turns, {s['updated_at'][:19]})")
            continue

        elif user_input.lower() == 'resume':
            sessions = store.list_sessions()
            if not sessions:
                print("  No conversations to resume.")
                continue
            print("\n  Select a conversation:")
            for i, s in enumerate(sessions):
                print(f"    {i + 1}. [{s['session_id']}] {s['title']} ({s['total_turns']} turns)")
            choice = input("  Enter number: ").strip()
            try:
                idx = int(choice) - 1
                current_session = sessions[idx]["session_id"]
                agent.resume_session(current_session)
                print(f"\n  Conversation resumed! Continue chatting.\n")
            except (ValueError, IndexError):
                print("  Invalid choice.")
            continue

        elif user_input.lower().startswith('search '):
            query = user_input[7:].strip()
            results = store.search_messages(query)
            if not results:
                print(f"  No results for '{query}'.")
            else:
                print(f"\n  🔍 Results for '{query}':")
                for r in results:
                    preview = r["content"][:100].replace("\n", " ")
                    print(f"    [{r['session_id']}] {r['title']} | {r['role']}: {preview}...")
            continue

        elif user_input.lower() == 'delete':
            sessions = store.list_sessions()
            if not sessions:
                print("  No conversations to delete.")
                continue
            for i, s in enumerate(sessions):
                print(f"    {i + 1}. [{s['session_id']}] {s['title']}")
            choice = input("  Delete which? ").strip()
            try:
                idx = int(choice) - 1
                sid = sessions[idx]["session_id"]
                store.delete_session(sid)
                if current_session == sid:
                    current_session = None
                print(f"  Deleted session {sid}.")
            except (ValueError, IndexError):
                print("  Invalid choice.")
            continue

        elif user_input.lower() == 'clear':
            confirm = input("  Delete ALL conversations? (yes/no): ").strip().lower()
            if confirm == 'yes':
                store.delete_all_sessions()
                current_session = None
                print("  All conversations deleted.")
            continue

        # --- Chat message ---
        if not current_session:
            print("  Start a conversation first: type 'new' or 'resume'")
            continue

        response = agent.chat(current_session, user_input)
        print(f"  AI: {response}")


# ============================================================
# PART 3: MongoDB Equivalent (Reference Implementation)
# ============================================================
# If you have MongoDB installed, this shows how the SAME pattern
# works with a production-grade database.
#
# Install: pip install pymongo
# Start MongoDB: mongod --dbpath /tmp/mongo_data
# ============================================================

MONGODB_EXAMPLE = """
# ─────────────────────────────────────────────────────────────
# MongoDB Conversation Store — Production Pattern
# ─────────────────────────────────────────────────────────────
# pip install pymongo
# This is the SAME pattern as SQLite above, just with MongoDB.
# MongoDB is preferred for production because:
#   - Better for high concurrency (many users at once)
#   - Native JSON storage (no schema migration headaches)
#   - Built-in TTL indexes (auto-delete old conversations)
#   - Horizontal scaling (sharding)
# ─────────────────────────────────────────────────────────────

from pymongo import MongoClient
from datetime import datetime
import uuid

class MongoConversationStore:
    def __init__(self, uri="mongodb://localhost:27017", db_name="ai_conversations"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.sessions = self.db["sessions"]
        self.messages = self.db["messages"]

        # Create indexes for fast lookups
        self.messages.create_index([("session_id", 1), ("created_at", 1)])
        self.messages.create_index([("content", "text")])  # Full-text search!

    def create_session(self, system_prompt, title=None, model="gpt-4o-mini"):
        session_id = str(uuid.uuid4())[:8]
        now = datetime.now()

        self.sessions.insert_one({
            "session_id": session_id,
            "title": title or f"Chat {now.strftime('%Y-%m-%d')}",
            "system_prompt": system_prompt,
            "model": model,
            "total_turns": 0,
            "created_at": now,
            "updated_at": now,
        })

        # Save system prompt as first message
        self.messages.insert_one({
            "session_id": session_id,
            "role": "system",
            "content": system_prompt,
            "created_at": now,
        })

        return session_id

    def get_messages(self, session_id):
        cursor = self.messages.find(
            {"session_id": session_id},
            {"_id": 0, "role": 1, "content": 1}
        ).sort("created_at", 1)
        return [{"role": m["role"], "content": m["content"]} for m in cursor]

    def add_message(self, session_id, role, content, token_count=0):
        now = datetime.now()
        self.messages.insert_one({
            "session_id": session_id,
            "role": role,
            "content": content,
            "token_count": token_count,
            "created_at": now,
        })
        self.sessions.update_one(
            {"session_id": session_id},
            {"$set": {"updated_at": now}, "$inc": {"total_turns": 1}}
        )

    def search_messages(self, query, limit=10):
        # MongoDB full-text search!
        return list(self.messages.find(
            {"$text": {"$search": query}},
            {"_id": 0, "session_id": 1, "role": 1, "content": 1}
        ).limit(limit))

    def list_sessions(self):
        return list(self.sessions.find(
            {}, {"_id": 0}
        ).sort("updated_at", -1))

    def delete_session(self, session_id):
        self.messages.delete_many({"session_id": session_id})
        self.sessions.delete_one({"session_id": session_id})
"""


def show_mongodb_reference():
    """Display the MongoDB equivalent implementation."""
    print("\n" + "=" * 70)
    print("  REFERENCE: MONGODB EQUIVALENT IMPLEMENTATION")
    print("=" * 70)
    print("""
  The SQLite pattern above is IDENTICAL to how production systems work.
  Below is the MongoDB equivalent — the SAME pattern, different DB.

  Why MongoDB for production?
  • High concurrency (many users chatting simultaneously)
  • Native JSON storage (conversation data maps naturally)
  • TTL indexes (auto-delete old conversations after 90 days)
  • Full-text search built-in
  • Horizontal scaling via sharding
  """)
    print(MONGODB_EXAMPLE)


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 5: DATABASE-BACKED CONVERSATION PERSISTENCE")
    print("=" * 70)
    print("""
    How do AI systems (ChatGPT, Copilot) remember your conversations?
    They store every message in a database, keyed by session ID.
    This exercise teaches you that exact pattern.

    Choose a demo:
      1 → Basic persistence — save, close, resume a conversation
      2 → Multiple conversations — like ChatGPT's sidebar
      3 → Search across all conversations
      4 → Interactive persistent chat (try save & resume!)
      5 → Show MongoDB reference implementation
      6 → Run demos 1-3 (automated)
    """)

    choice = input("  Enter choice (1-6): ").strip()

    if choice == "1":
        demo_basic_persistence()
    elif choice == "2":
        demo_multiple_sessions()
    elif choice == "3":
        demo_search_conversations()
    elif choice == "4":
        demo_interactive_chat()
    elif choice == "5":
        show_mongodb_reference()
    elif choice == "6":
        demo_basic_persistence()
        demo_multiple_sessions()
        demo_search_conversations()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. SESSION PERSISTENCE PATTERN: Every production AI chatbot uses
   this exact pattern:
     Load history from DB → Send to LLM → Save response to DB
   The LLM itself is stateless — the DATABASE is the memory.

2. SESSION IDs: Each conversation gets a unique ID (like ChatGPT's
   thread IDs). This lets you:
   - Run multiple independent conversations
   - Resume any conversation at any time
   - Share conversation links with others

3. WHY DATABASES (not files):
   - Concurrent access (multiple users)
   - ACID transactions (no corrupted saves)
   - Indexing (fast search across millions of messages)
   - TTL (auto-delete old conversations)
   - Backup and replication

4. SQLite vs MongoDB vs PostgreSQL:
   - SQLite: Perfect for single-user, local apps, prototyping
   - MongoDB: Great for JSON-heavy data, high concurrency
   - PostgreSQL: Best for complex queries, joins, analytics
   - Redis: Ultra-fast for session cache + TTL (often used as
     a cache layer IN FRONT of MongoDB/PostgreSQL)

5. CONTEXT WINDOW MANAGEMENT: As conversations grow long, you
   hit the LLM's context window limit. Production systems handle
   this by:
   - Truncating old messages (keep last N)
   - Summarizing old messages (compress history)
   - Using RAG to selectively retrieve relevant past messages
   
6. HOW COPILOT / CHATGPT LIKELY WORKS:
   - Messages stored in a distributed database (Cosmos DB, DynamoDB, etc.)
   - Session metadata cached in Redis for fast loading
   - Full-text search index (Elasticsearch) for searching across chats
   - Conversation history sent to LLM on each request
   - Old conversations auto-archived or summarized
"""


if __name__ == "__main__":
    main()
