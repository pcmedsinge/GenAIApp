"""
Demo: Understanding System Messages
Shows how system messages affect AI behavior
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def test_system_message(system_prompt: str, user_question: str):
    """Test how different system messages affect responses"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ],
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].message.content


# Same question, different system messages
question = "I am feeling very sleepy and continuously sweating. What should I do?"

print("="*70)
print("SYSTEM MESSAGE COMPARISON")
print("="*70)
print(f"\nUser Question: {question}\n")

# Test 1: Empathetic medical assistant
print("\n" + "="*70)
print("System Message 1: Empathetic Medical Intake Assistant")
print("="*70)
system1 = "You are a helpful medical intake assistant. Be empathetic and professional."
response1 = test_system_message(system1, question)
print(f"\nResponse:\n{response1}")

# Test 2: ER triage (urgent)
print("\n" + "="*70)
print("System Message 2: ER Triage Assistant")
print("="*70)
system2 = "You are an ER triage assistant. Be urgent and efficient. Ask critical questions first."
response2 = test_system_message(system2, question)
print(f"\nResponse:\n{response2}")

# Test 3: Medical educator (teaching)
print("\n" + "="*70)
print("System Message 3: Medical Educator")
print("="*70)
system3 = "You are a medical educator. Explain symptoms in educational terms with medical terminology."
response3 = test_system_message(system3, question)
print(f"\nResponse:\n{response3}")

# Test 4: Pediatric nurse (child-friendly)
print("\n" + "="*70)
print("System Message 4: Pediatric Nurse")
print("="*70)
system4 = "You are a pediatric nurse. Use simple, gentle language suitable for children and parents."
response4 = test_system_message(system4, question)
print(f"\nResponse:\n{response4}")

# Test 5: Combined instructions (RIGHT WAY)
print("\n" + "="*70)
print("System Message 5: Combined Instructions (CORRECT)")
print("="*70)
system5 = """You are an ER triage assistant with the following characteristics:
- Be empathetic but efficient
- Prioritize urgent symptoms
- Ask critical questions first
- Use professional medical terminology
- Keep responses concise"""
response5 = test_system_message(system5, question)
print(f"\nResponse:\n{response5}")

# Test 6: Multiple system messages (WRONG WAY - for comparison)
print("\n" + "="*70)
print("System Message 6: Multiple System Messages (WRONG)")
print("="*70)
print("Note: This uses multiple system messages - not recommended!")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Be empathetic and professional."},
        {"role": "system", "content": "Be urgent and efficient."},
        {"role": "system", "content": "Use simple language."},
        {"role": "user", "content": question}
    ],
    temperature=0.7,
    max_tokens=150
)
print(f"\nResponse:\n{response.choices[0].message.content}")
print("\n⚠️  The model may be confused by conflicting instructions!")

print("\n" + "="*70)
print("KEY TAKEAWAYS")
print("="*70)
print("""
1. ✅ USE ONE system message that combines all instructions
2. ❌ DON'T use multiple system messages (they conflict)
3. 💡 System message DRAMATICALLY affects tone, style, and content
4. 🎯 Be specific about the behavior you want
5. 📝 You can include multiple points in ONE system message
""")
