"""
Understanding the Response Object
Shows what OpenAI returns and how to access it
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def explore_response_structure():
    """
    Let's make a simple API call and inspect the response object
    """
    print("="*70)
    print("UNDERSTANDING THE RESPONSE OBJECT")
    print("="*70)
    
    # Make a simple API call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Say 'Hello'"}
        ],
        temperature=0.7,
        max_tokens=50
    )
    
    print("\n1. WHAT IS 'response'?")
    print("-" * 70)
    print(f"Type: {type(response)}")
    print(f"It's an object of type: {response.__class__.__name__}")
    
    print("\n2. FULL RESPONSE STRUCTURE:")
    print("-" * 70)
    # Convert to dict to see the structure
    response_dict = response.model_dump()
    print(json.dumps(response_dict, indent=2))
    
    print("\n3. BREAKING IT DOWN:")
    print("-" * 70)
    
    print("\n📋 response.id")
    print(f"   → {response.id}")
    print("   (Unique ID for this API call)")
    
    print("\n📋 response.model")
    print(f"   → {response.model}")
    print("   (Which model was actually used)")
    
    print("\n📋 response.created")
    print(f"   → {response.created}")
    print("   (Unix timestamp when created)")
    
    print("\n📋 response.object")
    print(f"   → {response.object}")
    print("   (Type of response object)")
    
    print("\n📋 response.choices (THIS IS THE ARRAY!)")
    print(f"   → Type: {type(response.choices)}")
    print(f"   → Length: {len(response.choices)}")
    print("   (List of possible responses - usually just 1)")
    
    print("\n4. WHY IS IT AN ARRAY? (choices)")
    print("-" * 70)
    print("""
The 'choices' is a LIST because:
- You can request multiple responses with parameter 'n'
- Each choice is a different possible answer
- Usually n=1 (default), so choices has only 1 item
- That's why we use choices[0] - get the first (and usually only) choice
    """)
    
    print("\n5. WHAT'S INSIDE choices[0]?")
    print("-" * 70)
    first_choice = response.choices[0]
    print(f"Type: {type(first_choice)}")
    print(f"\nStructure of choices[0]:")
    print(json.dumps(first_choice.model_dump(), indent=2))
    
    print("\n6. ACCESSING THE MESSAGE:")
    print("-" * 70)
    print(f"choices[0].index         → {first_choice.index}")
    print(f"choices[0].finish_reason → {first_choice.finish_reason}")
    print(f"choices[0].message       → {first_choice.message}")
    
    print("\n7. GETTING THE ACTUAL TEXT CONTENT:")
    print("-" * 70)
    message = first_choice.message
    print(f"message.role    → {message.role}")
    print(f"message.content → {message.content}")
    
    print("\n8. THE COMPLETE PATH:")
    print("-" * 70)
    print("""
response                          # The full API response object
  ↓
  .choices                        # List of possible responses
    ↓
    [0]                           # First (usually only) response
      ↓
      .message                    # The message object
        ↓
        .content                  # The actual text!

So: response.choices[0].message.content
    """)
    
    print("\n9. USAGE INFORMATION:")
    print("-" * 70)
    print(f"response.usage.prompt_tokens      → {response.usage.prompt_tokens}")
    print(f"response.usage.completion_tokens  → {response.usage.completion_tokens}")
    print(f"response.usage.total_tokens       → {response.usage.total_tokens}")


def multiple_choices_example():
    """
    Example showing why 'choices' is an array
    Using n=3 to get 3 different responses
    """
    print("\n\n" + "="*70)
    print("MULTIPLE CHOICES EXAMPLE (n=3)")
    print("="*70)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Give a one-word greeting"}
        ],
        temperature=1.0,  # Higher temperature for variety
        max_tokens=10,
        n=3  # Request 3 different responses!
    )
    
    print(f"\nNumber of choices returned: {len(response.choices)}")
    print("\nAll 3 responses:")
    
    for i, choice in enumerate(response.choices):
        print(f"\n  choices[{i}].message.content → {choice.message.content}")
    
    print("\n💡 See? 'choices' is an array because you can get multiple responses!")
    print("   But normally we just use n=1 (default) and access choices[0]")


def common_access_patterns():
    """
    Show common ways to access response data
    """
    print("\n\n" + "="*70)
    print("COMMON ACCESS PATTERNS")
    print("="*70)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "What is 2+2?"}
        ]
    )
    
    print("\n✅ CORRECT WAYS:")
    print("-" * 70)
    
    # Method 1: Direct access (most common)
    content = response.choices[0].message.content
    print(f"1. content = response.choices[0].message.content")
    print(f"   → {content}")
    
    # Method 2: Step by step (more readable)
    first_choice = response.choices[0]
    message = first_choice.message
    text = message.content
    print(f"\n2. Step by step:")
    print(f"   first_choice = response.choices[0]")
    print(f"   message = first_choice.message")
    print(f"   text = message.content")
    print(f"   → {text}")
    
    # Method 3: Get everything in a dict
    response_dict = response.model_dump()
    content = response_dict['choices'][0]['message']['content']
    print(f"\n3. Using dict (after model_dump()):")
    print(f"   content = response_dict['choices'][0]['message']['content']")
    print(f"   → {content}")
    
    print("\n\n❌ COMMON MISTAKES:")
    print("-" * 70)
    
    print("\n1. Forgetting [0]:")
    print("   ❌ response.choices.message.content")
    print("   ✅ response.choices[0].message.content")
    
    print("\n2. Treating it like a dict:")
    print("   ❌ response['choices'][0]['message']['content']")
    print("   ✅ response.choices[0].message.content")
    print("   (It's an object, not a dict! Use . not [])")
    
    print("\n3. Skipping .message:")
    print("   ❌ response.choices[0].content")
    print("   ✅ response.choices[0].message.content")


def main():
    """
    Run all examples
    """
    print("\n🔍 Understanding response.choices[0].message.content\n")
    
    # Explore the structure
    explore_response_structure()
    
    # Show multiple choices
    multiple_choices_example()
    
    # Common patterns
    common_access_patterns()
    
    print("\n\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
✅ KEY POINTS:

1. 'choices' is a LIST (array) of possible responses
2. We use [0] to get the FIRST response (usually the only one)
3. Each choice has a 'message' object
4. The message has 'content' which is the actual text
5. Full path: response.choices[0].message.content

WHY THIS STRUCTURE?
- Allows requesting multiple responses (n parameter)
- Consistent API structure
- Future-proof for additional fields
    """)


if __name__ == "__main__":
    main()
