"""
Project: Basic Chat with LLMs
Objective: Learn single-turn and multi-turn conversations
Concepts: Messages, roles, context, parameters

Healthcare Use Case: Patient symptom intake chatbot
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Anthropic only if API key is available
try:
    from anthropic import Anthropic
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_api_key and anthropic_api_key != "sk-ant-...":
        anthropic_client = Anthropic(api_key=anthropic_api_key)
        ANTHROPIC_AVAILABLE = True
    else:
        anthropic_client = None
        ANTHROPIC_AVAILABLE = False
except ImportError:
    anthropic_client = None
    ANTHROPIC_AVAILABLE = False


def simple_chat_openai(user_message: str) -> str:
    """
    Single-turn conversation with OpenAI
    
    Args:
        user_message: The user's question or statement
        
    Returns:
        The assistant's response
    """
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",  # Cost-effective model for learning
        messages=[
            # ONLY ONE system message - it sets the AI's behavior/personality
            {"role": "system", "content": "You are a helpful medical intake assistant. Be empathetic and professional."},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,  # 0.0 = deterministic, 2.0 = very creative
        max_tokens=500    # Limit response length
    )
    
    return response.choices[0].message.content


def simple_chat_claude(user_message: str) -> str:
    """
    Single-turn conversation with Claude
    
    Args:
        user_message: The user's question or statement
        
    Returns:
        The assistant's response
    """
    response = anthropic_client.messages.create(
        model="claude-3-5-haiku-20241022",  # Fast and cost-effective
        max_tokens=500,
        system="You are a helpful medical intake assistant. Be empathetic and professional.",
        messages=[
            {"role": "user", "content": user_message}
        ]
    )
    
    return response.content[0].text


def multi_turn_conversation(provider="openai"):
    """
    Multi-turn conversation with context retention
    Demonstrates how to maintain conversation history
    """
    print(f"\n{'='*60}")
    print(f"Medical Intake Assistant ({provider.upper()})")
    print(f"{'='*60}")
    print("I'll help you describe your symptoms. Type 'quit' to exit.\n")
    
    # System message defines the assistant's behavior
    system_message = """You are a medical intake assistant. Your job is to:
1. Gather patient symptoms professionally and empathetically
2. Ask clarifying questions about duration, severity, and triggers
3. DO NOT diagnose - just collect information
4. Summarize the information at the end
5. Keep responses concise (2-3 sentences)"""
    
    # Conversation history - maintains context
    messages = [
        {"role": "system", "content": system_message}
    ]
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nThank you. Take care!")
            break
            
        if not user_input:
            continue
        
        # Add user message to history
        messages.append({"role": "user", "content": user_input})
        
        try:
            if provider == "openai":
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=200
                )
                assistant_message = response.choices[0].message.content
                
            else:  # claude
                # Claude requires separating system message
                claude_messages = [m for m in messages if m["role"] != "system"]
                response = anthropic_client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=200,
                    system=system_message,
                    messages=claude_messages
                )
                assistant_message = response.content[0].text
            
            # Add assistant response to history
            messages.append({"role": "assistant", "content": assistant_message})
            
            print(f"\nAssistant: {assistant_message}\n")
            
        except Exception as e:
            print(f"Error: {e}")
            messages.pop()  # Remove the failed user message
            continue


def compare_parameters():
    """
    Demonstrates the effect of different parameters
    Shows the same prompt with different temperature settings
    """
    prompt = "Describe the symptoms of type 2 diabetes in simple terms."
    
    print("\n" + "="*60)
    print("PARAMETER COMPARISON: Temperature Effect")
    print("="*60)
    
    temperatures = [0.0, 0.7, 1.5]
    
    for temp in temperatures:
        print(f"\n--- Temperature: {temp} ---")
        if temp == 0.0:
            print("(Deterministic - same output every time)")
        elif temp < 1.0:
            print("(Balanced - good for most tasks)")
        else:
            print("(Creative - more variation)")
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a medical educator."},
                {"role": "user", "content": prompt}
            ],
            temperature=temp,
            max_tokens=150
        )
        
        print(f"\n{response.choices[0].message.content}\n")
        print("-" * 60)


def cost_tracking_example():
    """
    Demonstrates how to track token usage and costs
    Important for production applications!
    """
    print("\n" + "="*60)
    print("COST TRACKING EXAMPLE")
    print("="*60)
    
    prompt = "Explain what a CBC (Complete Blood Count) test measures."
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )
    
    # Extract usage information
    usage = response.usage
    
    print(f"\nPrompt: {prompt}")
    print(f"\nResponse: {response.choices[0].message.content}")
    print(f"\n{'='*60}")
    print("USAGE STATISTICS:")
    print(f"{'='*60}")
    print(f"Prompt tokens:     {usage.prompt_tokens}")
    print(f"Completion tokens: {usage.completion_tokens}")
    print(f"Total tokens:      {usage.total_tokens}")
    
    # Calculate approximate cost (prices as of 2024)
    # GPT-4o-mini: $0.150 per 1M input tokens, $0.600 per 1M output tokens
    input_cost = (usage.prompt_tokens / 1_000_000) * 0.150
    output_cost = (usage.completion_tokens / 1_000_000) * 0.600
    total_cost = input_cost + output_cost
    
    print(f"\nAPPROXIMATE COST:")
    print(f"Input:  ${input_cost:.6f}")
    print(f"Output: ${output_cost:.6f}")
    print(f"Total:  ${total_cost:.6f}")
    print(f"{'='*60}")


def main():
    """
    Run all examples
    Uncomment the ones you want to try
    """
    print("🏥 Level 1.1: Basic Chat Examples\n")
    
    # Example 1: Interactive single-turn chat (COMMENTED OUT)
    # print("1. SIMPLE CHAT (Single turn - Interactive)")
    # print("="*60)
    # print("Ask a medical question (or press Enter to skip):")
    # user_msg = input("You: ").strip()
    
    # if user_msg:
    #     print(f"\nOpenAI Response:\n{simple_chat_openai(user_msg)}")
        
    #     if ANTHROPIC_AVAILABLE:
    #         print(f"\nClaude Response:\n{simple_chat_claude(user_msg)}")
    # else:
    #     # Use default example
    #     user_msg = "I've been having headaches for the past week."
    #     print(f"\nUsing example: {user_msg}")
    #     print(f"\nOpenAI Response:\n{simple_chat_openai(user_msg)}")
        
    #     if ANTHROPIC_AVAILABLE:
    #         print(f"\nClaude Response:\n{simple_chat_claude(user_msg)}")
    #     else:
    #         print(f"\nClaude Response: [Skipped - Anthropic API key not configured]")
    
    # print("\n" + "="*60)
    # print("That was a SINGLE-TURN conversation.")
    # print("The AI has NO memory of what was just said.")
    # print("Each question is independent.\n")
    
    # Example 2: Multi-turn conversation (NOW ACTIVE!)
    # Interactive chat with memory - has conversation context
    # multi_turn_conversation(provider="openai")  # or "claude"
    
    # Example 3: Parameter comparison
    # Uncomment to see temperature effects
    compare_parameters()
    
    # Example 4: Cost tracking
    # Uncomment to see token usage
    # cost_tracking_example()


if __name__ == "__main__":
    main()
