"""
Exercise 1: Modify the System Prompt
Create different AI personalities through system messages

This demonstrates how system prompts dramatically change AI behavior
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================
# Different System Prompts / Personalities
# ============================================

SYSTEM_PROMPTS = {
    "pediatric_nurse": """You are a pediatric nurse speaking to a child and their parent. Your characteristics:
- Use simple, gentle language appropriate for children
- Be warm, friendly, and reassuring
- Avoid scary medical terms
- Use examples kids can understand (like comparing pain to "ouchies")
- Address both the child and parent
- Keep responses short and clear
- Use encouraging words""",
    
    "er_triage": """You are an ER triage assistant. Your role:
- Be urgent and efficient - time matters
- Ask critical questions FIRST (severity, onset, duration)
- Prioritize life-threatening symptoms
- Use medical terminology when appropriate
- Be direct and focused
- Assess priority level (immediate, urgent, standard)
- Keep responses very brief (1-2 sentences)""",
    
    "mental_health": """You are a mental health counselor. Your approach:
- Be deeply empathetic and supportive
- Create a safe, non-judgmental space
- Listen actively and validate feelings
- Use reflective listening techniques
- Ask open-ended questions gently
- Show genuine care and concern
- Be patient and give time for responses
- Focus on emotional wellbeing""",
    
    "medical_intake": """You are a general medical intake assistant. Your role:
- Be professional yet empathetic
- Gather comprehensive symptom information
- Ask about duration, severity, and patterns
- DO NOT diagnose
- Be thorough but efficient
- Suggest when to see a doctor"""
}


def chat_with_personality(personality: str, user_message: str) -> str:
    """
    Get response using specific personality/system prompt
    """
    system_prompt = SYSTEM_PROMPTS.get(personality, SYSTEM_PROMPTS["medical_intake"])
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_tokens=300
    )
    
    return response.choices[0].message.content


def compare_personalities():
    """
    Show how the same question gets different responses based on personality
    """
    # Test message - something a patient might say
    test_messages = [
        "I have a tummy ache and it hurts a lot",
        "I'm having chest discomfort and feel dizzy",
        "I've been feeling really anxious and can't sleep"
    ]
    
    print("="*70)
    print("EXERCISE 1: SYSTEM PROMPT PERSONALITIES")
    print("="*70)
    
    for test_msg in test_messages:
        print(f"\n{'='*70}")
        print(f"Patient says: '{test_msg}'")
        print(f"{'='*70}")
        
        # Get response from each personality
        for name, description in [
            ("pediatric_nurse", "Pediatric Nurse (gentle, child-friendly)"),
            ("er_triage", "ER Triage (urgent, efficient)"),
            ("mental_health", "Mental Health Counselor (empathetic)"),
            ("medical_intake", "Standard Medical Intake (balanced)")
        ]:
            print(f"\n🏥 {description}:")
            print("-" * 70)
            response = chat_with_personality(name, test_msg)
            print(response)
        
        print("\n")


def interactive_personality_selector():
    """
    Let user choose personality and have a conversation
    """
    print("\n" + "="*70)
    print("INTERACTIVE PERSONALITY SELECTOR")
    print("="*70)
    
    print("\nChoose an AI personality:")
    print("1. Pediatric Nurse (gentle, child-friendly)")
    print("2. ER Triage Assistant (urgent, efficient)")
    print("3. Mental Health Counselor (empathetic, supportive)")
    print("4. Standard Medical Intake (balanced)")
    
    choice = input("\nEnter number (1-4): ").strip()
    
    personality_map = {
        "1": "pediatric_nurse",
        "2": "er_triage",
        "3": "mental_health",
        "4": "medical_intake"
    }
    
    personality = personality_map.get(choice, "medical_intake")
    system_prompt = SYSTEM_PROMPTS[personality]
    
    print(f"\n✓ Selected: {personality.replace('_', ' ').title()}")
    print("\nType your message (or 'quit' to exit):\n")
    
    # Conversation with selected personality
    messages = [{"role": "system", "content": system_prompt}]
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nThank you!")
            break
        
        if not user_input:
            continue
        
        # Add user message
        messages.append({"role": "user", "content": user_input})
        
        # Get AI response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )
        
        assistant_message = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_message})
        
        print(f"\nAssistant: {assistant_message}\n")


def create_custom_personality():
    """
    Guide user to create their own custom system prompt
    """
    print("\n" + "="*70)
    print("CREATE YOUR OWN PERSONALITY")
    print("="*70)
    
    print("\nLet's create a custom AI personality!")
    print("Answer these questions:\n")
    
    role = input("1. What is the AI's role? (e.g., sports medicine doctor, nutritionist): ").strip()
    tone = input("2. What tone should it have? (e.g., friendly, formal, casual): ").strip()
    special = input("3. Any special characteristics? (e.g., uses analogies, asks follow-up questions): ").strip()
    
    # Build custom system prompt
    custom_prompt = f"""You are a {role}. Your communication style:
- Use a {tone} tone
- {special}
- Be helpful and professional
- Provide accurate information
- Ask clarifying questions when needed"""
    
    print("\n" + "="*70)
    print("YOUR CUSTOM SYSTEM PROMPT:")
    print("="*70)
    print(custom_prompt)
    print("="*70)
    
    # Test it
    test = input("\n\nTest your custom personality? (yes/no): ").strip().lower()
    if test in ['yes', 'y']:
        user_msg = input("\nYour message: ").strip()
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": custom_prompt},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        print(f"\nResponse:\n{response.choices[0].message.content}")


def main():
    """
    Run Exercise 1
    """
    print("\n🎭 Exercise 1: System Prompt Personalities\n")
    
    print("This exercise shows how system prompts create different AI personalities.\n")
    
    print("Choose an option:")
    print("1. Compare all personalities side-by-side")
    print("2. Interactive personality selector")
    print("3. Create your own custom personality")
    
    choice = input("\nEnter number (1-3): ").strip()
    
    if choice == "1":
        compare_personalities()
    elif choice == "2":
        interactive_personality_selector()
    elif choice == "3":
        create_custom_personality()
    else:
        print("\nRunning comparison by default...")
        compare_personalities()
    
    print("\n" + "="*70)
    print("KEY TAKEAWAY:")
    print("="*70)
    print("""
The system prompt is POWERFUL! It controls:
- Tone and style
- Level of detail
- Types of questions asked
- Language complexity
- Emotional approach

Same LLM, completely different behavior based on system prompt!
""")


if __name__ == "__main__":
    main()
