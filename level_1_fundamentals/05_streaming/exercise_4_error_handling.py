"""
Exercise 4: Error Handling
Handle network interruptions and partial responses during streaming.

Skills practiced:
- Robust error handling for streaming API calls
- Fallback from streaming to non-streaming
- Retry logic with exponential backofftion, different specialists)")
    print("4. Run demos 1 and 3")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        demo_single_question()
    elif choice == "2":
        demo_multi_turn_chat()
- Handling partial responses gracefully
- Timeout management

Healthcare context:
  In clinical settings, reliability matters. If a network hiccup interrupts
  a streaming clinical summary mid-sentence, your app can't just crash.
  It must recover gracefully — retry, fall back, or alert the user.
"""

import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================
# Strategy 1: Basic Try/Except with Fallback
# ============================================================

def stream_with_fallback(messages, max_tokens=300):
    """
    Try streaming first. If it fails, fall back to non-streaming.
    Returns: (content, method_used)
    """
    print("   📡 Attempting streaming response...")

    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=max_tokens,
            stream=True
        )

        full_content = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                token = chunk.choices[0].delta.content
                print(token, end="", flush=True)
                full_content += token

        print()
        return full_content, "streaming"

    except Exception as e:
        print(f"\n   ⚠️  Streaming failed: {e}")
        print("   🔄 Falling back to non-streaming...")

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=max_tokens,
                stream=False
            )
            content = response.choices[0].message.content
            print(content)
            return content, "fallback"

        except Exception as e2:
            print(f"   ❌ Fallback also failed: {e2}")
            return None, "failed"


# ============================================================
# Strategy 2: Retry with Exponential Backoff
# ============================================================

def stream_with_retry(messages, max_tokens=300, max_retries=3):
    """
    Retry streaming with exponential backoff.
    If all retries fail, fall back to non-streaming.
    """
    for attempt in range(1, max_retries + 1):
        try:
            print(f"   📡 Attempt {attempt}/{max_retries}...")

            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=max_tokens,
                stream=True
            )

            full_content = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    token = chunk.choices[0].delta.content
                    print(token, end="", flush=True)
                    full_content += token

            print()
            return full_content, attempt

        except Exception as e:
            wait_time = 2 ** attempt  # 2, 4, 8 seconds
            print(f"\n   ⚠️  Attempt {attempt} failed: {e}")

            if attempt < max_retries:
                print(f"   ⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"   ❌ All {max_retries} attempts failed.")

                # Final fallback
                try:
                    print("   🔄 Non-streaming fallback...")
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        max_tokens=max_tokens,
                        stream=False
                    )
                    content = response.choices[0].message.content
                    print(content)
                    return content, -1  # -1 indicates fallback
                except Exception:
                    return None, 0


# ============================================================
# Strategy 3: Chunk-Level Error Handling
# ============================================================

def stream_with_chunk_protection(messages, max_tokens=300):
    """
    Handle errors at the individual chunk level.
    Don't crash on a single bad chunk — keep going.
    """
    full_content = ""
    chunk_count = 0
    error_count = 0

    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=max_tokens,
            stream=True
        )

        for chunk in stream:
            try:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    token = chunk.choices[0].delta.content
                    print(token, end="", flush=True)
                    full_content += token
                    chunk_count += 1
            except (IndexError, AttributeError) as e:
                error_count += 1
                # Log but don't crash
                continue

        print()

    except Exception as e:
        print(f"\n   ❌ Stream-level error: {e}")
        if full_content:
            print(f"   📝 Partial content recovered ({len(full_content)} chars)")

    return full_content, chunk_count, error_count


# ============================================================
# Strategy 4: Partial Response Recovery
# ============================================================

def stream_with_partial_recovery(messages, max_tokens=300):
    """
    If stream is interrupted, save what we got and try to complete.
    Simulates handling partial responses.
    """
    full_content = ""
    chunk_count = 0

    print("   📡 Streaming (with partial recovery)...")

    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=max_tokens,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                token = chunk.choices[0].delta.content
                print(token, end="", flush=True)
                full_content += token
                chunk_count += 1

        print()
        return full_content, "complete"

    except Exception as e:
        print(f"\n\n   ⚠️  Stream interrupted after {chunk_count} chunks!")
        print(f"   📝 Partial content ({len(full_content)} chars):")
        print(f"      \"{full_content[:100]}...\"")

        if full_content:
            # Try to complete the response
            print(f"\n   🔄 Attempting to complete the interrupted response...")

            try:
                # Add partial response to context and ask to continue
                completion_messages = messages.copy()
                completion_messages.append({"role": "assistant", "content": full_content})
                completion_messages.append({"role": "user", "content": "Continue from where you left off. Do not repeat what was already said."})

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=completion_messages,
                    max_tokens=max_tokens,
                    stream=False
                )

                continuation = response.choices[0].message.content
                full_content += " " + continuation
                print(f"   ✅ Recovery successful! Added {len(continuation)} more chars.")
                print(f"\n   📝 CONTINUED:\n   {continuation}")

                return full_content, "recovered"

            except Exception as e2:
                print(f"   ❌ Recovery failed: {e2}")
                return full_content, "partial"
        else:
            return None, "failed"


# ============================================================
# Demo: Showcase All Error Handling Strategies
# ============================================================

def demo_all_strategies():
    """Run all strategies with working API calls"""
    messages = [
        {"role": "system", "content": "You are a medical assistant. Be concise."},
        {"role": "user", "content": "What are the 5 warning signs of a stroke? Use the FAST acronym plus one more."}
    ]

    print("\n" + "=" * 70)
    print("STRATEGY 1: STREAMING WITH FALLBACK")
    print("=" * 70)
    content, method = stream_with_fallback(messages)
    print(f"\n   ✅ Method used: {method}")

    print(f"\n{'═' * 70}")
    print("STRATEGY 2: RETRY WITH EXPONENTIAL BACKOFF")
    print("=" * 70)
    content, attempts = stream_with_retry(messages)
    print(f"\n   ✅ Succeeded on attempt: {attempts}")

    print(f"\n{'═' * 70}")
    print("STRATEGY 3: CHUNK-LEVEL PROTECTION")
    print("=" * 70)
    content, chunks, errors = stream_with_chunk_protection(messages)
    print(f"\n   ✅ Chunks: {chunks}, Errors caught: {errors}")

    print(f"\n{'═' * 70}")
    print("STRATEGY 4: PARTIAL RESPONSE RECOVERY")
    print("=" * 70)
    content, status = stream_with_partial_recovery(messages)
    print(f"\n   ✅ Status: {status}")


# ============================================================
# Demo: Simulated Failure Scenarios
# ============================================================

def demo_failure_scenarios():
    """Show how error handling works with intentional edge cases"""
    print("\n" + "=" * 70)
    print("FAILURE SCENARIO TESTS")
    print("=" * 70)

    # Scenario 1: Invalid model (will trigger fallback)
    print(f"\n{'─' * 70}")
    print("📋 Scenario 1: Invalid model → triggers error handling")
    print(f"{'─' * 70}")

    try:
        stream = client.chat.completions.create(
            model="gpt-nonexistent-model",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=50,
            stream=True
        )
        for chunk in stream:
            pass
    except Exception as e:
        print(f"   ❌ Caught error: {type(e).__name__}: {str(e)[:120]}")
        print("   ✅ This is expected! Your error handling should catch this.")

    # Scenario 2: Empty message triggers handling
    print(f"\n{'─' * 70}")
    print("📋 Scenario 2: Empty content handling")
    print(f"{'─' * 70}")

    content, chunks, errors = stream_with_chunk_protection([
        {"role": "user", "content": "Say just the word 'OK' and nothing else."}
    ], max_tokens=10)
    print(f"   ✅ Short response handled: {chunks} chunks, content: \"{content}\"")

    # Scenario 3: Very low max_tokens (truncated response)
    print(f"\n{'─' * 70}")
    print("📋 Scenario 3: Truncated response (max_tokens=20)")
    print(f"{'─' * 70}")
    print("   ", end="")
    content, status = stream_with_partial_recovery([
        {"role": "system", "content": "You are a medical educator."},
        {"role": "user", "content": "Explain heart failure in detail."}
    ], max_tokens=20)
    print(f"   ⚠️  Response was truncated by max_tokens limit.")
    print(f"   📝 Got: \"{content[:80]}...\"")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n🛡️ Exercise 4: Error Handling in Streaming")
    print("=" * 70)
    print("Build robust, production-ready streaming code\n")

    print("Choose a demo:")
    print("1. All 4 error handling strategies")
    print("2. Failure scenario tests")
    print("3. Interactive — test strategies with your own prompt")
    print("4. Run all demos")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1" or choice == "4":
        demo_all_strategies()

    if choice == "2" or choice == "4":
        demo_failure_scenarios()

    if choice == "3":
        print("\n💬 Enter a medical question to test error handling:")
        question = input("Question: ").strip()
        if question:
            messages = [
                {"role": "system", "content": "You are a medical assistant. Educational purposes only."},
                {"role": "user", "content": question}
            ]
            print(f"\n--- Streaming with fallback ---")
            stream_with_fallback(messages)
            print(f"\n--- Chunk-protected streaming ---")
            stream_with_chunk_protection(messages)

    if choice not in ["1", "2", "3", "4"]:
        print("Invalid choice")

    print(f"""
{'═' * 70}
KEY LEARNINGS:
{'═' * 70}

🛡️ ERROR HANDLING STRATEGIES:

   1. FALLBACK:  Stream fails → switch to non-streaming
      Best for: Simple apps, guaranteed delivery

   2. RETRY:     Fail → wait → try again (exponential backoff)
      Best for: Transient network errors, rate limits

   3. CHUNK PROTECTION:  Bad chunk → skip it, keep going
      Best for: Maximizing content recovery

   4. PARTIAL RECOVERY:  Interrupted → save partial → try to complete
      Best for: Long documents where losing progress is costly

🔑 PRODUCTION BEST PRACTICES:
   • ALWAYS wrap streaming in try/except
   • Log errors for monitoring (don't just swallow them)
   • Have a non-streaming fallback
   • Set reasonable timeouts
   • For clinical docs, save partial content rather than lose it
   • Retry rate limit errors (429) with backoff
   • Don't retry authentication errors (401)

🏥 HEALTHCARE RELIABILITY:
   Clinical systems need 99.9%+ uptime.
   Error handling is not optional — it's the difference between
   a production system and a demo.
""")


if __name__ == "__main__":
    main()
