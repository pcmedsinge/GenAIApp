"""
Exercise 4: Compare Models
Compare different LLM models side-by-side: quality, speed, and cost

This helps you make informed decisions about which model to use
"""

import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Anthropic only if available
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


# Model pricing (as of 2024-2025)
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.150, "output": 0.600, "per": 1_000_000},  # per 1M tokens
    "gpt-4o": {"input": 2.50, "output": 10.00, "per": 1_000_000},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00, "per": 1_000_000},
    "claude-3-5-haiku": {"input": 0.25, "output": 1.25, "per": 1_000_000},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00, "per": 1_000_000},
}


def test_openai_model(model: str, prompt: str, system_message: str = None) -> dict:
    """
    Test an OpenAI model and return metrics
    """
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    
    start_time = time.time()
    
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=300
    )
    
    end_time = time.time()
    
    # Calculate cost
    usage = response.usage
    pricing = MODEL_PRICING.get(model, {"input": 0, "output": 0, "per": 1_000_000})
    
    input_cost = (usage.prompt_tokens / pricing["per"]) * pricing["input"]
    output_cost = (usage.completion_tokens / pricing["per"]) * pricing["output"]
    total_cost = input_cost + output_cost
    
    return {
        "model": model,
        "response": response.choices[0].message.content,
        "time_seconds": round(end_time - start_time, 2),
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "cost_usd": round(total_cost, 6),
        "finish_reason": response.choices[0].finish_reason
    }


def test_claude_model(model: str, prompt: str, system_message: str = None) -> dict:
    """
    Test a Claude model and return metrics
    """
    if not ANTHROPIC_AVAILABLE:
        return {
            "model": model,
            "response": "[Claude not available - API key not configured]",
            "error": True
        }
    
    start_time = time.time()
    
    response = anthropic_client.messages.create(
        model=model,
        max_tokens=300,
        system=system_message or "You are a helpful assistant.",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    end_time = time.time()
    
    # Calculate cost
    pricing = MODEL_PRICING.get(model, {"input": 0, "output": 0, "per": 1_000_000})
    
    input_cost = (response.usage.input_tokens / pricing["per"]) * pricing["input"]
    output_cost = (response.usage.output_tokens / pricing["per"]) * pricing["output"]
    total_cost = input_cost + output_cost
    
    return {
        "model": model,
        "response": response.content[0].text,
        "time_seconds": round(end_time - start_time, 2),
        "prompt_tokens": response.usage.input_tokens,
        "completion_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        "cost_usd": round(total_cost, 6),
        "finish_reason": response.stop_reason
    }


def compare_models(prompt: str, system_message: str = None):
    """
    Compare multiple models side-by-side
    """
    print("="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(f"\nPrompt: {prompt}")
    if system_message:
        print(f"System: {system_message[:50]}...")
    print("\n" + "="*70)
    
    # Models to test
    models_to_test = [
        ("openai", "gpt-4o-mini"),
        ("openai", "gpt-4o"),
    ]
    
    if ANTHROPIC_AVAILABLE:
        models_to_test.extend([
            ("claude", "claude-3-5-haiku-20241022"),
            ("claude", "claude-3-5-sonnet-20241022"),
        ])
    
    results = []
    
    for provider, model in models_to_test:
        print(f"\n🤖 Testing {model}...")
        
        try:
            if provider == "openai":
                result = test_openai_model(model, prompt, system_message)
            else:
                result = test_claude_model(model, prompt, system_message)
            
            results.append(result)
            
            if result.get("error"):
                print(f"   ⚠️  {result['response']}")
            else:
                print(f"   ✓ Completed in {result['time_seconds']}s")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                "model": model,
                "error": str(e)
            })
    
    # Display results
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    for i, result in enumerate(results, 1):
        if result.get("error"):
            continue
        
        print(f"\n{i}. {result['model'].upper()}")
        print("-" * 70)
        print(f"Response:\n{result['response']}\n")
        print(f"⏱️  Time:   {result['time_seconds']}s")
        print(f"🎫 Tokens: {result['total_tokens']} ({result['prompt_tokens']} in, {result['completion_tokens']} out)")
        print(f"💰 Cost:   ${result['cost_usd']}")
        print(f"✓  Status: {result['finish_reason']}")
    
    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    
    valid_results = [r for r in results if not r.get("error")]
    
    if valid_results:
        print(f"\n{'Model':<30} {'Time (s)':<12} {'Tokens':<10} {'Cost ($)':<12}")
        print("-" * 70)
        
        for result in valid_results:
            print(f"{result['model']:<30} {result['time_seconds']:<12} {result['total_tokens']:<10} ${result['cost_usd']:<11.6f}")
        
        # Find best in each category
        fastest = min(valid_results, key=lambda x: x['time_seconds'])
        cheapest = min(valid_results, key=lambda x: x['cost_usd'])
        most_tokens = max(valid_results, key=lambda x: x['completion_tokens'])
        
        print("\n" + "="*70)
        print("🏆 WINNERS:")
        print("="*70)
        print(f"⚡ Fastest:       {fastest['model']} ({fastest['time_seconds']}s)")
        print(f"💰 Cheapest:      {cheapest['model']} (${cheapest['cost_usd']})")
        print(f"📝 Most detailed: {most_tokens['model']} ({most_tokens['completion_tokens']} tokens)")


def medical_use_case_comparison():
    """
    Compare models on a healthcare-specific task
    """
    print("\n" + "="*70)
    print("HEALTHCARE USE CASE: SYMPTOM ANALYSIS")
    print("="*70)
    
    system_message = """You are a medical triage assistant. Analyze the patient's symptoms and provide:
1. Severity assessment (mild/moderate/severe)
2. Recommended action (self-care/see doctor/urgent care/emergency)
3. Brief explanation"""
    
    test_cases = [
        "I have a mild headache that started this morning after waking up",
        "I'm experiencing chest pain and shortness of breath for the last 30 minutes",
        "My child has a fever of 102F and is complaining of ear pain"
    ]
    
    for test_case in test_cases:
        print(f"\n{'='*70}")
        print(f"Case: {test_case}")
        print(f"{'='*70}")
        
        compare_models(test_case, system_message)


def quality_comparison():
    """
    Compare models on response quality for complex questions
    """
    print("\n" + "="*70)
    print("QUALITY COMPARISON: COMPLEX MEDICAL QUESTION")
    print("="*70)
    
    prompt = """A 65-year-old patient with type 2 diabetes is taking metformin. 
They're asking if they can take ibuprofen for their arthritis pain. 
What should they know about this combination?"""
    
    system_message = "You are a pharmacist. Provide accurate drug interaction information."
    
    compare_models(prompt, system_message)


def cost_efficiency_test():
    """
    Test cost efficiency for bulk operations
    """
    print("\n" + "="*70)
    print("COST EFFICIENCY TEST (100 requests simulation)")
    print("="*70)
    
    prompt = "What are the symptoms of strep throat?"
    
    models = [
        ("gpt-4o-mini", "OpenAI"),
        ("claude-3-5-haiku-20241022", "Claude") if ANTHROPIC_AVAILABLE else None
    ]
    models = [m for m in models if m]  # Remove None
    
    print("\nSimulating 100 identical requests...\n")
    print(f"{'Model':<30} {'Cost per request':<20} {'Cost for 100':<20}")
    print("-" * 70)
    
    for model, provider in models:
        try:
            if provider == "OpenAI":
                result = test_openai_model(model, prompt)
            else:
                result = test_claude_model(model, prompt)
            
            if not result.get("error"):
                cost_per = result['cost_usd']
                cost_100 = cost_per * 100
                print(f"{model:<30} ${cost_per:<19.6f} ${cost_100:<19.2f}")
        except Exception as e:
            print(f"{model:<30} Error: {e}")


def main():
    """
    Run Exercise 4
    """
    print("\n⚖️  Exercise 4: Model Comparison\n")
    
    print("Choose comparison type:")
    print("1. Basic model comparison")
    print("2. Healthcare use case comparison")
    print("3. Quality comparison (complex question)")
    print("4. Cost efficiency test")
    print("5. Run all comparisons")
    
    choice = input("\nEnter number (1-5): ").strip()
    
    if choice == "1":
        prompt = "Explain the difference between a cold and the flu in simple terms."
        compare_models(prompt)
    
    elif choice == "2":
        medical_use_case_comparison()
    
    elif choice == "3":
        quality_comparison()
    
    elif choice == "4":
        cost_efficiency_test()
    
    elif choice == "5":
        prompt = "Explain the difference between a cold and the flu."
        compare_models(prompt)
        medical_use_case_comparison()
        cost_efficiency_test()
    
    else:
        print("\nRunning basic comparison...")
        prompt = "What are the early warning signs of diabetes?"
        compare_models(prompt)
    
    print("\n" + "="*70)
    print("KEY TAKEAWAYS:")
    print("="*70)
    print("""
📊 WHEN TO USE EACH MODEL:

gpt-4o-mini:
  ✅ Most cost-effective
  ✅ Fast responses
  ✅ Good for simple tasks, high volume
  ✅ Best for learning and development

gpt-4o:
  ✅ Higher quality responses
  ✅ Better at complex reasoning
  ✅ 10-20x more expensive
  ✅ Use for production, critical tasks

claude-3-5-haiku:
  ✅ Fast and cheap (like gpt-4o-mini)
  ✅ Good alternative to OpenAI
  ✅ Different "personality"

claude-3-5-sonnet:
  ✅ Very high quality
  ✅ Excellent at complex tasks
  ✅ More expensive (like gpt-4o)

💡 RECOMMENDATION:
- Development: gpt-4o-mini
- Production (simple): gpt-4o-mini or claude-haiku
- Production (complex): gpt-4o or claude-sonnet
- Always test with your specific use case!
""")


if __name__ == "__main__":
    main()
