"""
Exercise 1: Streaming Chat Interface
Build a streaming medical Q&A where responses appear word by word in real time.

Skills practiced:
- stream=True API parameter
- Iterating over chunks
- Collecting full response while displaying
- Multi-turn conversation with streaming

Healthcare context:
  Medical Q&A apps feel much more natural with streaming.
  Patients and clinicians see the answer forming in real time,
  just like ChatGPT — instead of waiting for the full response.
"""

import os
import sys
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================
# Core Streaming Function
# ============================================================

def stream_response(messages, model="gpt-4o-mini", max_tokens=500):
    """
    Stream a response and display it in real-time.
    Returns the full collected text and timing stats.
    """
    start_time = time.time()
    first_token_time = None
    full_content = ""
    chunk_count = 0

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                token = chunk.choices[0].delta.content

                if first_token_time is None:
                    first_token_time = time.time()

                # Print each token immediately
                print(token, end="", flush=True)
                full_content += token
                chunk_count += 1

        print()  # Newline after stream ends

    except Exception as e:
        print(f"\n❌ Stream error: {e}")
        return None, {}

    end_time = time.time()

    stats = {
        "first_token_ms": round((first_token_time - start_time) * 1000) if first_token_time else 0,
        "total_time_s": round(end_time - start_time, 2),
        "chunks": chunk_count,
        "characters": len(full_content),
    }

    return full_content, stats


# ============================================================
# Demo 1: Simple Streaming Medical Q&A
# ============================================================

def demo_single_question():
    """Stream a single medical question"""
    print("\n" + "=" * 70)
    print("DEMO 1: SINGLE QUESTION STREAMING")
    print("=" * 70)

    questions = [
        "Explain the ABCDE approach to emergency assessment in 150 words.",
        "What are the 5 most common drug interactions in elderly patients?",
        "Describe the difference between systolic and diastolic heart failure for a patient.",
    ]

    for q in questions:
        print(f"\n❓ {q}\n")
        print("💬 ", end="")
        content, stats = stream_response([
            {"role": "system", "content": "You are a medical educator. Be clear and concise. Educational purposes only."},
            {"role": "user", "content": q}
        ])
        print(f"\n   ⏱️  First token: {stats['first_token_ms']}ms | Total: {stats['total_time_s']}s | Chunks: {stats['chunks']}")
        print()


# ============================================================
# Demo 2: Multi-Turn Streaming Chat
# ============================================================

def demo_multi_turn_chat():
    """Interactive multi-turn chat with streaming responses"""
    print("\n" + "=" * 70)
    print("DEMO 2: MULTI-TURN STREAMING MEDICAL CHAT")
    print("=" * 70)
    print("""
💬 Have a conversation with the medical assistant!
   The assistant remembers the full conversation.
   Watch responses appear in real-time.
   Type 'quit' to exit.
""")

    messages = [
        {"role": "system", "content": """You are a knowledgeable medical education assistant.
- Provide accurate, evidence-based information
- Use clear language appropriate for healthcare professionals
- Remember the conversation context for follow-up questions
- Include disclaimers for clinical information
- Keep responses focused and practical"""}
    ]

    turn_count = 0

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Session ended.")
            break
        if not user_input:
            continue

        turn_count += 1
        messages.append({"role": "user", "content": user_input})

        print(f"\nAssistant: ", end="")
        content, stats = stream_response(messages)

        if content:
            messages.append({"role": "assistant", "content": content})
            print(f"   [{stats['first_token_ms']}ms to first token, {stats['chunks']} chunks]")
        print()

    print(f"\n📊 Session stats: {turn_count} turns, {len(messages)-1} messages in history")


# ============================================================
# Demo 3: Streaming with Different Personas
# ============================================================

def demo_personas():
    """Same question, different streaming personas"""
    print("\n" + "=" * 70)
    print("DEMO 3: STREAMING WITH DIFFERENT PERSONAS")
    print("=" * 70)

    question = "A patient asks: 'My doctor said I have high cholesterol. What does that mean?'"

    personas = {
        "Cardiologist": "You are a cardiologist explaining cholesterol to a patient. Be thorough but not scary.",
        "Pediatric Nurse": "You are a pediatric nurse. Explain cholesterol using simple, friendly language a teenager could understand.",
        "Pharmacist": "You are a pharmacist. Focus on what cholesterol means for medications and lifestyle.",
    }

    for name, system_prompt in personas.items():
        print(f"\n{'─' * 70}")
        print(f"🩺 {name}:")
        print(f"{'─' * 70}")
        content, stats = stream_response([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ], max_tokens=250)
        print(f"   [{stats['total_time_s']}s, {stats['chunks']} chunks]")

    print(f"""
💡 OBSERVATION:
   Same question, different streaming responses based on persona.
   The system prompt shapes not just WHAT is said but HOW it's said.
   Streaming makes each persona feel like a real conversation!
""")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n💬 Exercise 1: Streaming Chat Interface")
    print("=" * 70)
    print("Build real-time medical Q&A with streaming\n")

    print("Choose a demo:")
    print("1. Single question streaming (preset medical questions)")
    print("2. Multi-turn streaming chat (interactive)")
    print("3. Streaming personas (same question, different specialists)")
    print("4. Run demos 1 and 3")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        demo_single_question()
    elif choice == "2":
        demo_multi_turn_chat()
    elif choice == "3":
        demo_personas()
    elif choice == "4":
        demo_single_question()
        demo_personas()
    else:
        print("Invalid choice")

    print(f"""
{'═' * 70}
KEY LEARNINGS:
{'═' * 70}
1. stream=True makes responses appear in real-time (like ChatGPT)
2. flush=True in print() is essential — forces immediate display
3. Collect full_content for history while streaming for display
4. First-token time is usually 200-500ms (user sees text FAST)
5. Multi-turn streaming needs careful message history management
""")


if __name__ == "__main__":
    main()
