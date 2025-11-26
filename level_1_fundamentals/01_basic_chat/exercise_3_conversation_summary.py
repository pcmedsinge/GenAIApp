"""
Exercise 3: Conversation Summary
Automatically summarize conversations and extract key information

This is crucial for healthcare: documenting encounters, extracting symptoms
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def summarize_conversation(conversation_history: list) -> dict:
    """
    Use LLM to summarize a conversation and extract key information
    
    Args:
        conversation_history: List of message dicts with 'role' and 'content'
        
    Returns:
        Dict with summary, symptoms, and recommendations
    """
    # Build conversation text for the LLM to analyze
    conversation_text = ""
    for msg in conversation_history:
        if msg["role"] == "user":
            conversation_text += f"Patient: {msg['content']}\n"
        elif msg["role"] == "assistant":
            conversation_text += f"Assistant: {msg['content']}\n"
    
    # System prompt for summarization
    summary_prompt = """You are a medical documentation assistant. Analyze the conversation and provide:

1. SUMMARY: A concise 2-3 sentence summary of the conversation
2. KEY SYMPTOMS: List all symptoms mentioned with severity/duration if stated
3. MEDICAL PROFESSIONAL: Suggest which type of doctor should be consulted (e.g., primary care, cardiologist, psychiatrist, emergency room)
4. URGENCY: Rate urgency as LOW, MEDIUM, HIGH, or EMERGENCY
5. ADDITIONAL NOTES: Any other relevant information

Format your response EXACTLY like this:

SUMMARY:
[Your summary here]

KEY SYMPTOMS:
- [Symptom 1]
- [Symptom 2]

RECOMMENDED SPECIALIST:
[Type of doctor]

URGENCY LEVEL:
[LOW/MEDIUM/HIGH/EMERGENCY]

ADDITIONAL NOTES:
[Any other relevant info]"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": summary_prompt},
            {"role": "user", "content": f"Analyze this medical intake conversation:\n\n{conversation_text}"}
        ],
        temperature=0.3,  # Lower temperature for consistent formatting
        max_tokens=500
    )
    
    summary_text = response.choices[0].message.content
    
    return {
        "full_summary": summary_text,
        "conversation_text": conversation_text
    }


def conversation_with_summary():
    """
    Have a conversation, then automatically generate summary at the end
    """
    print("="*70)
    print("MEDICAL INTAKE WITH AUTO-SUMMARY")
    print("="*70)
    print("\nI'll help gather your symptoms. Type 'done' when finished.\n")
    
    system_message = """You are a medical intake assistant. Your job is to:
1. Gather patient symptoms professionally and empathetically
2. Ask about duration, severity, triggers, and associated symptoms
3. DO NOT diagnose - just collect information
4. Keep responses concise (2-3 sentences)
5. After 3-4 exchanges, suggest wrapping up"""
    
    messages = [{"role": "system", "content": system_message}]
    
    exchange_count = 0
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['done', 'quit', 'exit']:
            break
        
        if not user_input:
            continue
        
        messages.append({"role": "user", "content": user_input})
        exchange_count += 1
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )
        
        assistant_message = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_message})
        
        print(f"\nAssistant: {assistant_message}\n")
    
    # Generate summary
    if exchange_count > 0:
        print("\n" + "="*70)
        print("GENERATING CONVERSATION SUMMARY...")
        print("="*70)
        
        # Remove system message for summary
        conversation_only = [m for m in messages if m["role"] != "system"]
        
        summary = summarize_conversation(conversation_only)
        
        print("\n" + "="*70)
        print("CONVERSATION SUMMARY")
        print("="*70)
        print(summary["full_summary"])
        print("="*70)
    else:
        print("\nNo conversation to summarize.")


def demo_summary_examples():
    """
    Show examples of summarizing different types of conversations
    """
    print("="*70)
    print("DEMO: CONVERSATION SUMMARY EXAMPLES")
    print("="*70)
    
    # Example conversations
    examples = [
        {
            "title": "Headache Case",
            "conversation": [
                {"role": "user", "content": "I've been having really bad headaches for the past week"},
                {"role": "assistant", "content": "I'm sorry to hear that. Can you describe where the pain is and how severe it is on a scale of 1-10?"},
                {"role": "user", "content": "It's mostly behind my eyes, like a 7 or 8. Really painful."},
                {"role": "assistant", "content": "That sounds quite severe. Have you noticed any triggers, like screen time or certain foods?"},
                {"role": "user", "content": "Now that you mention it, it gets worse after I've been on my computer for a while"},
                {"role": "assistant", "content": "Screen time can definitely contribute. Have you experienced any other symptoms like nausea, visual changes, or sensitivity to light?"},
                {"role": "user", "content": "Yes, sometimes I feel nauseous and lights bother me"}
            ]
        },
        {
            "title": "Chest Pain Case",
            "conversation": [
                {"role": "user", "content": "I'm having chest pain and I'm worried"},
                {"role": "assistant", "content": "I understand your concern. Can you describe the pain - is it sharp, dull, or pressure-like? And when did it start?"},
                {"role": "user", "content": "It's like pressure on my chest, started about an hour ago"},
                {"role": "assistant", "content": "Is the pain radiating to your arm, jaw, or back? Are you experiencing shortness of breath or sweating?"},
                {"role": "user", "content": "My left arm feels a bit tingly and I am sweating"}
            ]
        },
        {
            "title": "Mental Health Case",
            "conversation": [
                {"role": "user", "content": "I've been feeling really down lately and can't seem to enjoy anything"},
                {"role": "assistant", "content": "I'm sorry you're going through this. How long have you been feeling this way?"},
                {"role": "user", "content": "Maybe 3 or 4 weeks now. I just don't want to do anything."},
                {"role": "assistant", "content": "Have you noticed changes in your sleep, appetite, or energy levels?"},
                {"role": "user", "content": "Yeah, I'm sleeping way more than usual but still tired. Not hungry either."}
            ]
        }
    ]
    
    for example in examples:
        print(f"\n{'='*70}")
        print(f"EXAMPLE: {example['title']}")
        print(f"{'='*70}")
        
        # Show conversation
        print("\nConversation:")
        print("-" * 70)
        for msg in example["conversation"]:
            role = "Patient" if msg["role"] == "user" else "Assistant"
            print(f"{role}: {msg['content']}")
        
        # Generate and show summary
        summary = summarize_conversation(example["conversation"])
        
        print(f"\n{'-'*70}")
        print("AUTOMATIC SUMMARY:")
        print(f"{'-'*70}")
        print(summary["full_summary"])


def extract_structured_data():
    """
    Show how to extract structured data from conversation
    """
    print("\n" + "="*70)
    print("EXTRACTING STRUCTURED DATA")
    print("="*70)
    
    conversation = [
        {"role": "user", "content": "I've had a fever of 101F for 3 days, along with a cough and body aches"},
        {"role": "assistant", "content": "I see. Is the cough dry or producing mucus? And are you experiencing any shortness of breath?"},
        {"role": "user", "content": "It's a dry cough, pretty annoying. No trouble breathing though."}
    ]
    
    # Extract as structured JSON
    extract_prompt = """Extract the following information from the conversation in JSON format:
{
  "symptoms": ["list of symptoms"],
  "duration": "how long symptoms have been present",
  "severity": "mild/moderate/severe",
  "vital_signs": {"temperature": "value if mentioned", "other": "any other vitals"},
  "associated_symptoms": ["related symptoms"]
}

Only include information that was explicitly mentioned."""
    
    conversation_text = "\n".join([
        f"{'Patient' if m['role']=='user' else 'Assistant'}: {m['content']}"
        for m in conversation
    ])
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": extract_prompt},
            {"role": "user", "content": conversation_text}
        ],
        temperature=0.1,
        max_tokens=300
    )
    
    print("\nConversation:")
    print(conversation_text)
    print("\n" + "-"*70)
    print("Extracted Structured Data:")
    print("-"*70)
    print(response.choices[0].message.content)


def main():
    """
    Run Exercise 3
    """
    print("\n📋 Exercise 3: Conversation Summary\n")
    
    print("Choose an option:")
    print("1. Have a conversation with auto-summary")
    print("2. See demo examples with summaries")
    print("3. Extract structured data from conversation")
    
    choice = input("\nEnter number (1-3): ").strip()
    
    if choice == "1":
        conversation_with_summary()
    elif choice == "2":
        demo_summary_examples()
    elif choice == "3":
        extract_structured_data()
    else:
        print("\nRunning demo examples...")
        demo_summary_examples()
    
    print("\n" + "="*70)
    print("KEY TAKEAWAYS:")
    print("="*70)
    print("""
✅ LLMs are excellent at summarization
✅ Can extract key information automatically
✅ Useful for documentation and record-keeping
✅ Can format as text or structured data (JSON)
✅ Critical for healthcare: accurate documentation

In production:
- Store original conversation + summary
- Have human review critical summaries
- Use for quality assurance
- Feed into EHR systems
""")


if __name__ == "__main__":
    main()
