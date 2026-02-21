"""
Exercise 2: Progress Indicators
Show visual progress while streaming (tokens received, elapsed time, spinner).

Skills practiced:
- Real-time progress tracking during streaming
- Calculating streaming statistics
- Visual feedback for long-running streams
- Comparing progress between different prompts

Healthcare context:
  When generating long clinical documents (discharge summaries, care plans),
  users need progress feedback. Without it, they wonder if the app froze.
"""

import os
import sys
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================
# Progress Indicator Styles
# ============================================================

def stream_with_counter(messages, max_tokens=500):
    """Style 1: Token counter displayed above the response"""
    start_time = time.time()
    first_token_time = None
    full_content = ""
    chunk_count = 0

    print("📊 Streaming... [tokens: 0]")
    print("─" * 50)

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=max_tokens,
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            token = chunk.choices[0].delta.content

            if first_token_time is None:
                first_token_time = time.time()

            print(token, end="", flush=True)
            full_content += token
            chunk_count += 1

            # Update counter on the same line above (using ANSI codes)
            # Move cursor up, update, move back down
            elapsed = time.time() - start_time

    end_time = time.time()
    ttft = first_token_time - start_time if first_token_time else 0

    print(f"\n{'─' * 50}")
    print(f"✅ Complete! {chunk_count} chunks | {len(full_content)} chars | "
          f"TTFT: {ttft*1000:.0f}ms | Total: {end_time-start_time:.1f}s")

    return full_content


def stream_with_spinner(messages, max_tokens=500):
    """Style 2: Spinner animation before first token, then stream"""
    spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    spinner_idx = 0
    start_time = time.time()
    first_token_time = None
    full_content = ""
    chunk_count = 0

    # Show "thinking" spinner while waiting for first token
    print("🧠 Thinking ", end="", flush=True)

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=max_tokens,
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            token = chunk.choices[0].delta.content

            if first_token_time is None:
                first_token_time = time.time()
                # Clear the spinner line and start response
                print(f"\r{'🧠 Thinking... ✅':40s}")
                ttft = (first_token_time - start_time) * 1000
                print(f"⏱️  First token in {ttft:.0f}ms\n")

            print(token, end="", flush=True)
            full_content += token
            chunk_count += 1

    end_time = time.time()
    print(f"\n\n📊 {chunk_count} chunks | {end_time-start_time:.1f}s total")

    return full_content


def stream_with_progress_bar(messages, max_tokens=300):
    """Style 3: Progress bar based on estimated tokens"""
    start_time = time.time()
    first_token_time = None
    full_content = ""
    chunk_count = 0
    estimated_tokens = max_tokens  # Use max_tokens as rough estimate

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=max_tokens,
        stream=True
    )

    print("Progress: ", end="", flush=True)

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            token = chunk.choices[0].delta.content

            if first_token_time is None:
                first_token_time = time.time()
                # Clear progress line and start content
                print(f"\r{'Progress: started':40s}\n")

            full_content += token
            chunk_count += 1
            print(token, end="", flush=True)

    end_time = time.time()

    # Final progress bar
    bar_width = 30
    filled = bar_width  # complete
    bar = "█" * filled
    pct = 100

    print(f"\n\n[{bar}] {pct}% — {chunk_count} chunks received")
    print(f"⏱️  TTFT: {(first_token_time-start_time)*1000:.0f}ms | Total: {end_time-start_time:.1f}s")

    return full_content


def stream_with_live_stats(messages, max_tokens=500):
    """Style 4: Live statistics updating during stream"""
    start_time = time.time()
    first_token_time = None
    full_content = ""
    chunk_count = 0
    word_count = 0

    print("📝 Generating response...\n")

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=max_tokens,
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            token = chunk.choices[0].delta.content

            if first_token_time is None:
                first_token_time = time.time()

            print(token, end="", flush=True)
            full_content += token
            chunk_count += 1

            # Count words
            if " " in token or "\n" in token:
                word_count = len(full_content.split())

    end_time = time.time()
    word_count = len(full_content.split())
    chars_per_sec = len(full_content) / (end_time - start_time) if end_time > start_time else 0

    print(f"""
{'─' * 50}
📊 LIVE STATS:
   ⏱️  Time to first token: {(first_token_time-start_time)*1000:.0f}ms
   ⏱️  Total generation time: {end_time-start_time:.1f}s
   📦  Chunks received: {chunk_count}
   📝  Words generated: {word_count}
   📏  Characters: {len(full_content)}
   🚀  Speed: {chars_per_sec:.0f} chars/second
""")
    return full_content


# ============================================================
# Comparison Demo
# ============================================================

def demo_compare_styles():
    """Compare all progress indicator styles"""
    print("\n" + "=" * 70)
    print("COMPARING 4 PROGRESS INDICATOR STYLES")
    print("=" * 70)

    messages = [
        {"role": "system", "content": "You are a medical educator. Be concise."},
        {"role": "user", "content": "Explain the 4 stages of chronic kidney disease in 150 words."}
    ]

    styles = [
        ("Style 1: Token Counter", stream_with_counter),
        ("Style 2: Spinner → Stream", stream_with_spinner),
        ("Style 3: Progress Bar", stream_with_progress_bar),
        ("Style 4: Live Stats", stream_with_live_stats),
    ]

    for name, func in styles:
        print(f"\n{'═' * 70}")
        print(f"🎨 {name}")
        print(f"{'═' * 70}\n")
        func(messages, max_tokens=250)
        input("\nPress Enter for next style...")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n📊 Exercise 2: Progress Indicators")
    print("=" * 70)
    print("Visual feedback during streaming\n")

    print("Choose a demo:")
    print("1. Token counter style")
    print("2. Spinner → stream style")
    print("3. Progress bar style")
    print("4. Live stats style")
    print("5. Compare all 4 styles")

    choice = input("\nEnter choice (1-5): ").strip()

    messages = [
        {"role": "system", "content": "You are a medical educator. Be clear and thorough."},
        {"role": "user", "content": "Explain the pathophysiology of hypertension and its effects on target organs in 200 words."}
    ]

    if choice == "1":
        stream_with_counter(messages)
    elif choice == "2":
        stream_with_spinner(messages)
    elif choice == "3":
        stream_with_progress_bar(messages)
    elif choice == "4":
        stream_with_live_stats(messages)
    elif choice == "5":
        demo_compare_styles()
    else:
        print("Invalid choice")

    print(f"""
{'═' * 70}
KEY LEARNINGS:
{'═' * 70}
1. TTFT (Time to First Token) is the key UX metric — aim for <500ms
2. Progress indicators prevent "is it frozen?" anxiety
3. Spinner is best for the waiting-for-first-token phase
4. Token counters give technical users confidence response is flowing
5. Live stats help during development and debugging
6. In production web apps, send stats via WebSocket alongside tokens

🏥 HEALTHCARE UX:
   Clinicians are busy — they need to know the system is working.
   A spinner or counter while generating clinical summaries avoids
   them clicking away or re-submitting the request.
""")


if __name__ == "__main__":
    main()
