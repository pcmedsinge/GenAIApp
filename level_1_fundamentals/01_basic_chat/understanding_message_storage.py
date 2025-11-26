"""
Understanding Message Storage in Multi-Turn Conversations
Shows WHERE and HOW conversation history is maintained
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def single_turn_no_memory():
    """
    Single turn - NO memory storage
    Each call is independent
    """
    print("="*70)
    print("SINGLE-TURN: NO MEMORY")
    print("="*70)
    
    # First question
    print("\n1st Question: 'I have a headache'")
    response1 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "I have a headache"}
        ]
    )
    print(f"Response: {response1.choices[0].message.content[:100]}...")
    
    # Second question - AI has NO memory of first question
    print("\n2nd Question: 'Where is the pain?'")
    response2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Where is the pain?"}  # AI doesn't know about headache!
        ]
    )
    print(f"Response: {response2.choices[0].message.content[:100]}...")
    print("\n❌ AI has NO CONTEXT - doesn't know about the headache!")


def multi_turn_with_memory():
    """
    Multi-turn - WITH memory storage
    We manually maintain conversation history
    """
    print("\n\n" + "="*70)
    print("MULTI-TURN: WITH MEMORY (Python List)")
    print("="*70)
    
    # THIS IS WHERE MESSAGES ARE STORED - A PYTHON LIST!
    messages = [
        {"role": "system", "content": "You are a medical assistant."}
    ]
    
    print("\n📦 Initial messages list:")
    print(messages)
    
    # First exchange
    print("\n" + "-"*70)
    print("1st Exchange:")
    print("-"*70)
    
    user_msg_1 = "I have a headache"
    print(f"User: {user_msg_1}")
    
    # ADD user message to list
    messages.append({"role": "user", "content": user_msg_1})
    print(f"\n📦 After adding user message, messages = ")
    print(messages)
    
    # Send entire conversation history to API
    response1 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages  # ← SEND THE WHOLE LIST
    )
    
    assistant_msg_1 = response1.choices[0].message.content
    print(f"\nAssistant: {assistant_msg_1[:100]}...")
    
    # ADD assistant response to list
    messages.append({"role": "assistant", "content": assistant_msg_1})
    print(f"\n📦 After adding assistant message, messages = ")
    for i, msg in enumerate(messages):
        role = msg['role']
        content = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
        print(f"  [{i}] {role}: {content}")
    
    # Second exchange
    print("\n" + "-"*70)
    print("2nd Exchange:")
    print("-"*70)
    
    user_msg_2 = "Where is the pain located?"
    print(f"User: {user_msg_2}")
    
    # ADD second user message
    messages.append({"role": "user", "content": user_msg_2})
    print(f"\n📦 After adding 2nd user message, messages = ")
    for i, msg in enumerate(messages):
        role = msg['role']
        content = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
        print(f"  [{i}] {role}: {content}")
    
    # Send ENTIRE conversation history again (including headache context!)
    response2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages  # ← SEND THE WHOLE LIST (now has 4 items)
    )
    
    assistant_msg_2 = response2.choices[0].message.content
    print(f"\nAssistant: {assistant_msg_2[:150]}...")
    
    # ADD second assistant response
    messages.append({"role": "assistant", "content": assistant_msg_2})
    
    print(f"\n📦 Final messages list has {len(messages)} items:")
    for i, msg in enumerate(messages):
        role = msg['role']
        content = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
        print(f"  [{i}] {role}: {content}")
    
    print("\n✅ AI REMEMBERS - because we sent the entire conversation history!")


def show_api_calls():
    """
    Show exactly what gets sent to the API each time
    """
    print("\n\n" + "="*70)
    print("WHAT ACTUALLY GETS SENT TO THE API")
    print("="*70)
    
    messages = [{"role": "system", "content": "You are helpful."}]
    
    print("\n1️⃣  FIRST API CALL:")
    print("-"*70)
    messages.append({"role": "user", "content": "My name is John"})
    print("Sending to API:")
    for msg in messages:
        print(f"  {msg}")
    
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=50)
    assistant_msg = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_msg})
    print(f"\nGot back: {assistant_msg}")
    
    print("\n2️⃣  SECOND API CALL:")
    print("-"*70)
    messages.append({"role": "user", "content": "What's my name?"})
    print("Sending to API (ENTIRE HISTORY!):")
    for msg in messages:
        content = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
        print(f"  {msg['role']}: {content}")
    
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=50)
    assistant_msg = response.choices[0].message.content
    print(f"\nGot back: {assistant_msg}")
    
    print("\n💡 See? We send the ENTIRE conversation every time!")
    print("   The API itself has NO memory - we provide the context!")


def memory_storage_locations():
    """
    Explain WHERE conversation memory can be stored
    """
    print("\n\n" + "="*70)
    print("WHERE TO STORE CONVERSATION MEMORY")
    print("="*70)
    
    print("""
1. 🐍 IN-MEMORY (Python List) - What we've been doing
   
   messages = []  # ← Stored in RAM
   
   ✅ Simple and fast
   ❌ Lost when program ends
   ❌ Lost when app restarts
   
   Use for: Quick demos, testing, short sessions

2. 💾 DATABASE (Persistent Storage)
   
   # Save to PostgreSQL, MongoDB, etc.
   db.conversations.insert({
       "user_id": "patient_123",
       "messages": messages,
       "timestamp": datetime.now()
   })
   
   ✅ Survives restarts
   ✅ Can be retrieved later
   ✅ Multiple users/sessions
   
   Use for: Production apps, multi-session conversations

3. 📁 FILE SYSTEM (JSON/Text Files)
   
   import json
   with open(f"conversation_{user_id}.json", "w") as f:
       json.dump(messages, f)
   
   ✅ Simple persistence
   ✅ Easy to inspect
   ❌ Doesn't scale well
   
   Use for: Development, logging, auditing

4. 🔐 SESSION STORAGE (Web Apps)
   
   session['messages'] = messages  # Flask/Django
   
   ✅ Per-user isolation
   ✅ Automatic cleanup
   ❌ Lost when session expires
   
   Use for: Web applications, temporary conversations

5. 🧠 REDIS/CACHE (Fast Access)
   
   redis.set(f"chat:{user_id}", json.dumps(messages))
   
   ✅ Very fast
   ✅ Shared across servers
   ❌ More complex setup
   
   Use for: High-traffic production apps
""")


def demonstrate_message_list_operations():
    """
    Show common operations on the messages list
    """
    print("\n\n" + "="*70)
    print("COMMON MESSAGE LIST OPERATIONS")
    print("="*70)
    
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Message 1"},
        {"role": "assistant", "content": "Response 1"},
        {"role": "user", "content": "Message 2"},
        {"role": "assistant", "content": "Response 2"},
        {"role": "user", "content": "Message 3"},
        {"role": "assistant", "content": "Response 3"},
    ]
    
    print(f"\n📦 Original list has {len(messages)} messages")
    
    # 1. Keep only recent messages (limit context)
    print("\n1️⃣  Keep only last 4 messages + system:")
    system = messages[0]
    recent = messages[-4:]
    limited = [system] + recent
    print(f"   Now has {len(limited)} messages (save tokens!)")
    
    # 2. Remove a failed message
    print("\n2️⃣  Remove last message (if there was an error):")
    messages_copy = messages.copy()
    removed = messages_copy.pop()
    print(f"   Removed: {removed['role']}")
    print(f"   Now has {len(messages_copy)} messages")
    
    # 3. Get only user messages
    print("\n3️⃣  Extract all user messages:")
    user_only = [m for m in messages if m['role'] == 'user']
    print(f"   Found {len(user_only)} user messages")
    
    # 4. Count tokens (approximate)
    print("\n4️⃣  Estimate total tokens:")
    total_chars = sum(len(m['content']) for m in messages)
    approx_tokens = total_chars // 4  # Rough estimate
    print(f"   ~{approx_tokens} tokens (4 chars ≈ 1 token)")
    
    # 5. Clear old messages but keep system
    print("\n5️⃣  Start fresh conversation (keep system):")
    fresh = [messages[0]]  # Keep system message only
    print(f"   Reset to {len(fresh)} message(s)")


def main():
    """
    Run all examples
    """
    print("\n💾 Understanding Message Storage in Multi-Turn Conversations\n")
    
    # Show the difference
    single_turn_no_memory()
    multi_turn_with_memory()
    
    # Show API calls
    show_api_calls()
    
    # Where to store
    memory_storage_locations()
    
    # Operations
    demonstrate_message_list_operations()
    
    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
✅ Messages are stored in a PYTHON LIST (in your code)
✅ The list is a simple array of dictionaries
✅ Each message has 'role' and 'content'
✅ You send the ENTIRE list with every API call
✅ The API itself has NO memory
✅ YOU are responsible for managing conversation history

STRUCTURE:
messages = [
    {"role": "system", "content": "..."},      # [0] System (instructions)
    {"role": "user", "content": "..."},        # [1] User's first message
    {"role": "assistant", "content": "..."},   # [2] AI's first response
    {"role": "user", "content": "..."},        # [3] User's second message
    {"role": "assistant", "content": "..."},   # [4] AI's second response
    # ... keeps growing with each exchange
]

IMPORTANT:
- More messages = more tokens = higher cost
- Limit history for long conversations
- Store in database for production
- Always include system message at start
""")


if __name__ == "__main__":
    main()
