"""
Exercise 2: Input Validation and Emergency Detection
Shows how to detect medical emergencies and add safety disclaimers
"""

import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================
# Emergency Detection
# ============================================

# Keywords that might indicate medical emergencies
EMERGENCY_KEYWORDS = [
    # Cardiac/Chest
    "chest pain", "heart attack", "crushing chest", "chest pressure",
    "left arm pain", "jaw pain", "severe chest",
    
    # Neurological
    "stroke", "can't move", "face drooping", "slurred speech",
    "severe headache", "worst headache", "thunderclap headache",
    "loss of consciousness", "passed out", "fainting",
    
    # Respiratory
    "can't breathe", "difficulty breathing", "shortness of breath",
    "choking", "severe asthma", "blue lips",
    
    # Trauma
    "severe bleeding", "heavy bleeding", "spurting blood",
    "broken bone", "head injury", "car accident",
    
    # Other Critical
    "suicide", "kill myself", "overdose", "poisoning",
    "severe abdominal pain", "pregnant and bleeding",
    "anaphylaxis", "allergic reaction", "swollen throat"
]


def detect_emergency(user_input: str) -> dict:
    """
    Detect if user input contains emergency keywords
    
    Returns:
        dict with 'is_emergency' (bool) and 'matched_keywords' (list)
    """
    user_input_lower = user_input.lower()
    matched = []
    
    for keyword in EMERGENCY_KEYWORDS:
        if keyword in user_input_lower:
            matched.append(keyword)
    
    return {
        "is_emergency": len(matched) > 0,
        "matched_keywords": matched
    }


def show_emergency_warning(matched_keywords: list):
    """Display emergency warning to user"""
    print("\n" + "!"*70)
    print("⚠️  EMERGENCY DETECTED ⚠️")
    print("!"*70)
    print(f"\nDetected emergency keywords: {', '.join(matched_keywords)}")
    print("\n🚨 IF THIS IS A MEDICAL EMERGENCY:")
    print("   - United States: CALL 911 IMMEDIATELY")
    print("   - UK: CALL 999")
    print("   - Europe: CALL 112")
    print("   - Or go to the nearest Emergency Room")
    print("\n⚠️  DO NOT RELY ON THIS CHATBOT FOR EMERGENCIES!")
    print("!"*70 + "\n")


def show_disclaimer():
    """Show disclaimer at the start"""
    print("\n" + "="*70)
    print("MEDICAL DISCLAIMER")
    print("="*70)
    print("""
This is an AI chatbot for informational purposes only.

⚠️  IMPORTANT:
- This is NOT a substitute for professional medical advice
- This is NOT for medical emergencies
- Always consult a licensed healthcare provider
- If you have a medical emergency, call 911 (or your local emergency number)

By continuing, you acknowledge you understand these limitations.
""")
    print("="*70 + "\n")


def validate_input(user_input: str) -> bool:
    """
    Validate user input
    Returns True if input is acceptable, False otherwise
    """
    # Check if empty
    if not user_input or not user_input.strip():
        print("⚠️  Please enter a valid message.")
        return False
    
    # Check if too short (likely not a real question)
    if len(user_input.strip()) < 3:
        print("⚠️  Please enter a more detailed message.")
        return False
    
    # Check if too long (prevent abuse)
    if len(user_input) > 1000:
        print("⚠️  Message too long. Please keep it under 1000 characters.")
        return False
    
    return True


def safe_chat_response(user_message: str) -> str:
    """
    Chat function with safety features built in
    """
    # System message with safety instructions
    system_message = """You are a medical information chatbot. Your role:

CRITICAL SAFETY RULES:
1. NEVER diagnose medical conditions
2. ALWAYS recommend seeing a doctor for specific health concerns
3. If you detect emergency symptoms, tell user to call 911 immediately
4. Remind users this is informational only, not medical advice
5. Be helpful but always emphasize limitations

Your responses should:
- Provide general health information
- Suggest when to see a doctor
- Be empathetic but cautious
- Include disclaimers when appropriate"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_tokens=300
    )
    
    return response.choices[0].message.content


# ============================================
# Main Interactive Function
# ============================================

def interactive_safe_chat():
    """
    Interactive chat with all safety features
    """
    # Show disclaimer first
    show_disclaimer()
    
    print("Type your health question (or 'quit' to exit):\n")
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        # Check for exit
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nThank you. Remember to consult healthcare professionals for medical advice!")
            break
        
        # Validate input
        if not validate_input(user_input):
            continue
        
        # Check for emergency keywords
        emergency_check = detect_emergency(user_input)
        
        if emergency_check["is_emergency"]:
            # Show emergency warning
            show_emergency_warning(emergency_check["matched_keywords"])
            
            # Ask if they want to continue
            cont = input("Do you still want to continue? (yes/no): ").strip().lower()
            if cont not in ['yes', 'y']:
                print("\nPlease seek immediate medical attention. Stay safe!")
                break
        
        # Get AI response
        print("\nAssistant: ", end="")
        response = safe_chat_response(user_input)
        print(response)
        print("\n" + "-"*70 + "\n")


# ============================================
# Demo Examples
# ============================================

def demo_emergency_detection():
    """
    Demo showing emergency detection
    """
    print("\n" + "="*70)
    print("DEMO: Emergency Detection")
    print("="*70)
    
    test_inputs = [
        "I have chest pain and my left arm hurts",
        "I've been having headaches lately",
        "I think I'm having a heart attack",
        "What are the symptoms of diabetes?",
        "I can't breathe properly",
        "I have a mild cough"
    ]
    
    for test_input in test_inputs:
        print(f"\nTest Input: '{test_input}'")
        result = detect_emergency(test_input)
        
        if result["is_emergency"]:
            print(f"  ⚠️  EMERGENCY DETECTED!")
            print(f"  Matched: {result['matched_keywords']}")
        else:
            print(f"  ✓ No emergency detected")
    
    print("\n" + "="*70)


def demo_validation():
    """
    Demo showing input validation
    """
    print("\n" + "="*70)
    print("DEMO: Input Validation")
    print("="*70)
    
    test_inputs = [
        "",  # Empty
        "hi",  # Too short
        "x" * 1100,  # Too long
        "I have a headache that started yesterday"  # Valid
    ]
    
    for i, test_input in enumerate(test_inputs, 1):
        display = test_input[:50] + "..." if len(test_input) > 50 else test_input
        print(f"\n{i}. Testing: '{display}'")
        result = validate_input(test_input)
        print(f"   Valid: {result}")
    
    print("\n" + "="*70)


# ============================================
# Main
# ============================================

def main():
    """
    Run the exercise
    """
    print("\n🏥 Exercise 2: Input Validation & Emergency Detection\n")
    
    # Show demos first
    demo_emergency_detection()
    demo_validation()
    
    print("\n" + "="*70)
    print("Ready to try the interactive safe chat?")
    print("="*70)
    choice = input("\nStart interactive chat? (yes/no): ").strip().lower()
    
    if choice in ['yes', 'y']:
        interactive_safe_chat()
    else:
        print("\nDemo completed. Run this script again to try interactive chat!")


if __name__ == "__main__":
    main()
