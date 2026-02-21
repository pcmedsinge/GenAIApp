"""
Project 5: Streaming Responses
Objective: Learn to stream LLM responses in real-time
Concepts: Streaming API, chunked responses, time-to-first-token, error handling

Healthcare Use Case: Real-time medical Q&A, clinical documentation
"""

import os
import time
import sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================
# DEMO 1: Streaming vs Non-Streaming
# ============================================

def demo_no_streaming():
    """
    Without streaming — user waits for entire response
    """
    print("\n" + "="*70)
    print("DEMO 1a: WITHOUT STREAMING")
    print("="*70)
    print("→ You will wait until the ENTIRE response is ready...\n")

    start_time = time.time()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a medical educator."},
            {"role": "user", "content": "Explain the pathophysiology of heart failure in 200 words."}
        ],
        max_tokens=300
    )

    end_time = time.time()
    content = response.choices[0].message.content

    print(f"⏳ Waited {end_time - start_time:.2f} seconds for complete response\n")
    print(content)
    print(f"\n📊 Total tokens: {response.usage.total_tokens}")
    print(f"   Time to see ANYTHING: {end_time - start_time:.2f}s (had to wait for ALL of it)")


def demo_with_streaming():
    """
    With streaming — user sees text appear in real-time
    """
    print("\n\n" + "="*70)
    print("DEMO 1b: WITH STREAMING")
    print("="*70)
    print("→ Watch the text appear word by word!\n")

    start_time = time.time()
    first_token_time = None
    full_content = ""
    token_count = 0

    # stream=True is the key difference!
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a medical educator."},
            {"role": "user", "content": "Explain the pathophysiology of heart failure in 200 words."}
        ],
        max_tokens=300,
        stream=True  # ← THIS enables streaming!
    )

    # Process each chunk as it arrives
    for chunk in stream:
        # Each chunk has a delta (the new piece of text)
        if chunk.choices[0].delta.content is not None:
            token_text = chunk.choices[0].delta.content

            # Record time of first token
            if first_token_time is None:
                first_token_time = time.time()

            # Print each token immediately (no newline, flush buffer)
            print(token_text, end="", flush=True)

            full_content += token_text
            token_count += 1

    end_time = time.time()

    print(f"\n\n📊 Streaming Stats:")
    print(f"   Time to FIRST token: {first_token_time - start_time:.3f}s ← User sees text almost instantly!")
    print(f"   Total time: {end_time - start_time:.2f}s")
    print(f"   Chunks received: {token_count}")
    print(f"   Collected {len(full_content)} characters")


# ============================================
# DEMO 2: Understanding Chunk Structure
# ============================================

def demo_chunk_structure():
    """
    See what each streaming chunk looks like
    """
    print("\n\n" + "="*70)
    print("DEMO 2: CHUNK STRUCTURE (What's inside each chunk?)")
    print("="*70)
    print("→ See the raw data as it arrives\n")

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "List 3 symptoms of pneumonia."}
        ],
        max_tokens=100,
        stream=True
    )

    print("Each chunk looks like this:\n")

    for i, chunk in enumerate(stream):
        delta = chunk.choices[0].delta
        finish_reason = chunk.choices[0].finish_reason

        if i < 10:  # Show first 10 chunks in detail
            print(f"  Chunk {i}:")
            print(f"    delta.role    = {delta.role}")
            print(f"    delta.content = {repr(delta.content)}")
            print(f"    finish_reason = {finish_reason}")
            print()

        if finish_reason == "stop":
            print(f"  Chunk {i}: finish_reason = 'stop' ← DONE!")
            break

    print("""
💡 KEY POINTS:
   - First chunk has role='assistant', content=None
   - Middle chunks have content='word' or content=' word'
   - Last chunk has finish_reason='stop'
   - delta.content is None when there's no new text
""")


# ============================================
# DEMO 3: Streaming Medical Q&A Chat
# ============================================

def demo_streaming_chat():
    """
    Interactive streaming medical chat
    """
    print("\n\n" + "="*70)
    print("DEMO 3: STREAMING MEDICAL Q&A CHAT")
    print("="*70)

    messages = [
        {"role": "system", "content": """You are a knowledgeable medical assistant.
Provide clear, accurate medical information.
Always include a disclaimer that this is for educational purposes only.
Use simple language that patients can understand."""}
    ]

    print("\n💬 Ask medical questions! Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye! Remember to consult your healthcare provider.")
            break

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        print("\nAssistant: ", end="", flush=True)

        # Stream the response
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500,
            stream=True
        )

        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                token = chunk.choices[0].delta.content
                print(token, end="", flush=True)
                full_response += token

        print("\n")  # New line after response

        # Add assistant's response to history
        messages.append({"role": "assistant", "content": full_response})


# ============================================
# DEMO 4: Stream with Progress Indicator
# ============================================

def demo_streaming_with_progress():
    """
    Streaming with a token counter and timing info
    """
    print("\n\n" + "="*70)
    print("DEMO 4: STREAMING WITH PROGRESS INDICATOR")
    print("="*70)

    questions = [
        "Explain the difference between Type 1 and Type 2 diabetes in 150 words.",
        "What are the 5 warning signs of stroke? Be concise.",
    ]

    for question in questions:
        print(f"\n📝 Question: {question}\n")
        print("=" * 70)

        start_time = time.time()
        first_token_time = None
        token_count = 0
        full_content = ""

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a medical educator. Be concise and clear."},
                {"role": "user", "content": question}
            ],
            max_tokens=300,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                token = chunk.choices[0].delta.content

                if first_token_time is None:
                    first_token_time = time.time()

                print(token, end="", flush=True)
                full_content += token
                token_count += 1

        end_time = time.time()

        # Print stats
        print(f"\n{'─' * 70}")
        print(f"⏱️  First token: {(first_token_time - start_time)*1000:.0f}ms | "
              f"Total: {end_time - start_time:.2f}s | "
              f"Chunks: {token_count} | "
              f"Chars: {len(full_content)}")
        print()


# ============================================
# DEMO 5: Stream Processing (Keyword Detection)
# ============================================

def demo_stream_processing():
    """
    Process streamed text to detect keywords as they appear
    """
    print("\n\n" + "="*70)
    print("DEMO 5: REAL-TIME STREAM PROCESSING")
    print("="*70)
    print("→ Detect emergency keywords AS the response streams in\n")

    emergency_keywords = [
        "emergency", "911", "immediately", "life-threatening",
        "call ambulance", "chest pain", "stroke", "bleeding",
        "unconscious", "not breathing", "seizure", "anaphylaxis"
    ]

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a medical triage assistant."},
            {"role": "user", "content": "A 60-year-old is having sudden chest pain and difficulty breathing. What should be done?"}
        ],
        max_tokens=400,
        stream=True
    )

    full_content = ""
    detected_keywords = []

    print("Response (keywords highlighted):\n")

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            token = chunk.choices[0].delta.content
            full_content += token

            # Check for emergency keywords in accumulated text
            for keyword in emergency_keywords:
                if keyword in full_content.lower() and keyword not in detected_keywords:
                    detected_keywords.append(keyword)
                    # Print token with alert
                    print(token, end="", flush=True)
                    print(f" 🚨", end="", flush=True)
                    break
            else:
                print(token, end="", flush=True)

    print(f"\n\n{'─' * 70}")
    if detected_keywords:
        print(f"🚨 EMERGENCY KEYWORDS DETECTED: {', '.join(detected_keywords)}")
        print(f"⚠️  This response contains urgent medical advice!")
    else:
        print("✅ No emergency keywords detected.")


# ============================================
# DEMO 6: Error Handling in Streams
# ============================================

def demo_error_handling():
    """
    How to handle errors during streaming
    """
    print("\n\n" + "="*70)
    print("DEMO 6: ERROR HANDLING IN STREAMS")
    print("="*70)

    print("\n--- Safe streaming with error handling ---\n")

    def safe_stream(messages, max_tokens=300):
        """Stream with proper error handling"""
        full_content = ""

        try:
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=max_tokens,
                stream=True
            )

            for chunk in stream:
                try:
                    if chunk.choices[0].delta.content is not None:
                        token = chunk.choices[0].delta.content
                        print(token, end="", flush=True)
                        full_content += token
                except (IndexError, AttributeError) as e:
                    print(f"\n⚠️  Chunk processing error: {e}")
                    continue

            print()  # New line after stream ends
            return full_content

        except Exception as e:
            print(f"\n❌ Streaming error: {e}")
            print("Falling back to non-streaming...")

            # Fallback to non-streaming
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=max_tokens,
                    stream=False
                )
                content = response.choices[0].message.content
                print(content)
                return content
            except Exception as e2:
                print(f"❌ Fallback also failed: {e2}")
                return None

    # Test safe streaming
    result = safe_stream([
        {"role": "user", "content": "What is normal blood pressure range? Be brief."}
    ])

    if result:
        print(f"\n✅ Successfully received {len(result)} characters")

    print("""
💡 ERROR HANDLING BEST PRACTICES:
   1. Wrap stream in try/except
   2. Handle individual chunk errors (don't crash on one bad chunk)
   3. Have a non-streaming fallback
   4. Log errors for monitoring
   5. Set timeouts for production apps
""")


# ============================================
# Main Menu
# ============================================

def main():
    print("\n⚡ Project 5: Streaming Responses")
    print("="*70)
    print("Learn to stream LLM responses for real-time user experience")

    print("\n\nChoose a demo:")
    print("1. Streaming vs Non-Streaming comparison")
    print("2. Chunk structure (what's inside each chunk)")
    print("3. Streaming medical Q&A chat")
    print("4. Streaming with progress indicators")
    print("5. Real-time keyword detection in stream")
    print("6. Error handling in streams")
    print("7. Run ALL demos (except chat)")

    choice = input("\nEnter choice (1-7): ").strip()

    if choice == "1":
        demo_no_streaming()
        demo_with_streaming()
    elif choice == "2":
        demo_chunk_structure()
    elif choice == "3":
        demo_streaming_chat()
    elif choice == "4":
        demo_streaming_with_progress()
    elif choice == "5":
        demo_stream_processing()
    elif choice == "6":
        demo_error_handling()
    elif choice == "7":
        demo_no_streaming()
        demo_with_streaming()
        demo_chunk_structure()
        demo_streaming_with_progress()
        demo_stream_processing()
        demo_error_handling()
    else:
        print("❌ Invalid choice")

    print("\n\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
⚡ STREAMING BASICS:
   - Add stream=True to your API call
   - Iterate over chunks with a for loop
   - Each chunk has delta.content (the new text)
   - Last chunk has finish_reason='stop'

📊 PERFORMANCE:
   - Time to FIRST token: ~200-500ms (user sees text fast!)
   - Without streaming: Wait 2-10s for entire response
   - Total time is roughly the same — but UX is much better

🏥 HEALTHCARE USE CASES:
   - Real-time patient Q&A (feels conversational)
   - Clinical documentation (see notes forming live)
   - Emergency keyword detection (alert as keywords appear)
   - Education content delivery (engaging reading experience)

⚠️  ERROR HANDLING:
   - Always wrap streams in try/except
   - Have non-streaming fallback
   - Handle individual chunk errors
   - Set timeouts in production

🔑 WHEN TO STREAM:
   ✅ User-facing chat interfaces
   ✅ Long responses (>100 tokens)
   ✅ Real-time monitoring/detection
   ❌ JSON output (need complete response to parse)
   ❌ Background processing (no user watching)
   ❌ Function calling (tools don't stream)

🎉 CONGRATULATIONS!
   You've completed Level 1: GenAI Fundamentals!
   You now understand APIs, embeddings, function calling,
   prompt engineering, and streaming.
   
   Ready for Level 2: RAG Systems! 🚀
""")


if __name__ == "__main__":
    main()
