"""
Exercise 4: Cost Comparison - Embeddings vs Full LLM
Calculate and compare costs of different approaches
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import time

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def calculate_embedding_cost(num_tokens, model="text-embedding-3-small"):
    """
    Calculate cost of embedding API calls
    
    Pricing (as of 2024):
    - text-embedding-3-small: $0.02 per 1M tokens
    - text-embedding-3-large: $0.13 per 1M tokens
    """
    pricing = {
        "text-embedding-3-small": 0.02 / 1_000_000,
        "text-embedding-3-large": 0.13 / 1_000_000,
    }
    
    return num_tokens * pricing[model]


def calculate_llm_cost(num_tokens, model="gpt-4o-mini"):
    """
    Calculate cost of LLM API calls
    
    Pricing (as of 2024):
    - gpt-4o-mini: $0.150 per 1M input tokens, $0.600 per 1M output tokens
    - gpt-4o: $2.50 per 1M input tokens, $10.00 per 1M output tokens
    """
    pricing = {
        "gpt-4o-mini": {
            "input": 0.150 / 1_000_000,
            "output": 0.600 / 1_000_000,
        },
        "gpt-4o": {
            "input": 2.50 / 1_000_000,
            "output": 10.00 / 1_000_000,
        },
    }
    
    # For similarity tasks, assume average 100 tokens output per call
    avg_output_tokens = 100
    
    input_cost = num_tokens * pricing[model]["input"]
    output_cost = avg_output_tokens * pricing[model]["output"]
    
    return input_cost + output_cost


def estimate_tokens(text):
    """
    Rough estimation: ~4 characters per token
    More accurate: use tiktoken library
    """
    return len(text) // 4


def demo_basic_cost_comparison():
    """
    Demo 1: Basic cost comparison for 10,000 medical terms
    """
    print("\n" + "="*70)
    print("DEMO 1: Cost Comparison - 10,000 Medical Terms")
    print("="*70)
    
    # Typical medical term length
    avg_term_length = 50  # characters
    num_terms = 10_000
    
    total_characters = avg_term_length * num_terms
    total_tokens = total_characters // 4
    
    print(f"\n📊 Scenario:")
    print(f"   - Number of medical terms: {num_terms:,}")
    print(f"   - Average term length: {avg_term_length} characters")
    print(f"   - Total characters: {total_characters:,}")
    print(f"   - Estimated tokens: {total_tokens:,}")
    
    print(f"\n💰 EMBEDDING APPROACH:")
    print("-"*70)
    
    # Embeddings - one-time cost to create embeddings
    small_embed_cost = calculate_embedding_cost(total_tokens, "text-embedding-3-small")
    large_embed_cost = calculate_embedding_cost(total_tokens, "text-embedding-3-large")
    
    print(f"   text-embedding-3-small: ${small_embed_cost:.4f}")
    print(f"   text-embedding-3-large: ${large_embed_cost:.4f}")
    print(f"\n   ✅ One-time cost to create embeddings")
    print(f"   ✅ Then FREE similarity comparisons (local computation)")
    print(f"   ✅ Can do UNLIMITED searches after initial embedding")
    
    print(f"\n💰 LLM APPROACH (for comparison):")
    print("-"*70)
    
    # If we used LLM for each similarity check
    num_comparisons = 100  # typical: 100 searches per day
    
    mini_llm_cost = calculate_llm_cost(total_tokens * num_comparisons, "gpt-4o-mini")
    full_llm_cost = calculate_llm_cost(total_tokens * num_comparisons, "gpt-4o")
    
    print(f"   For {num_comparisons} similarity checks:")
    print(f"   gpt-4o-mini: ${mini_llm_cost:.2f}")
    print(f"   gpt-4o: ${full_llm_cost:.2f}")
    print(f"\n   ❌ Ongoing cost for EVERY search")
    print(f"   ❌ Much more expensive for repeated searches")
    
    print(f"\n📈 SAVINGS:")
    print("-"*70)
    savings_mini = mini_llm_cost - small_embed_cost
    savings_full = full_llm_cost - small_embed_cost
    
    print(f"   vs gpt-4o-mini: ${savings_mini:.2f} saved ({savings_mini/mini_llm_cost*100:.1f}% reduction)")
    print(f"   vs gpt-4o: ${savings_full:.2f} saved ({savings_full/full_llm_cost*100:.1f}% reduction)")


def demo_real_use_case():
    """
    Demo 2: Real healthcare use case cost analysis
    """
    print("\n\n" + "="*70)
    print("DEMO 2: Real Use Case - ICD-10 Code Suggestion System")
    print("="*70)
    
    # Scenario: Hospital with ICD-10 code suggestion
    num_icd_codes = 70_000  # Total ICD-10 codes
    avg_description_length = 100  # characters per code
    
    # Usage
    encounters_per_day = 500
    days_per_month = 30
    total_encounters_per_month = encounters_per_day * days_per_month
    
    print(f"\n🏥 Hospital System:")
    print(f"   - ICD-10 codes in database: {num_icd_codes:,}")
    print(f"   - Patient encounters per day: {encounters_per_day:,}")
    print(f"   - Monthly encounters: {total_encounters_per_month:,}")
    
    # Calculate tokens
    total_tokens = (num_icd_codes * avg_description_length) // 4
    
    print(f"\n💰 APPROACH 1: Embeddings (RECOMMENDED)")
    print("-"*70)
    
    # One-time embedding cost
    initial_embed_cost = calculate_embedding_cost(total_tokens, "text-embedding-3-small")
    
    # Monthly maintenance (new codes added)
    new_codes_per_month = 100
    monthly_new_embed_cost = calculate_embedding_cost(
        (new_codes_per_month * avg_description_length) // 4,
        "text-embedding-3-small"
    )
    
    print(f"   Initial setup: ${initial_embed_cost:.2f} (one-time)")
    print(f"   Monthly updates: ${monthly_new_embed_cost:.4f}")
    print(f"   Searching: $0.00 (free local computation)")
    print(f"\n   📊 Total first month: ${initial_embed_cost + monthly_new_embed_cost:.2f}")
    print(f"   📊 Each subsequent month: ${monthly_new_embed_cost:.4f}")
    
    print(f"\n💰 APPROACH 2: LLM per query (NOT RECOMMENDED)")
    print("-"*70)
    
    # Assume each encounter needs LLM call
    avg_note_tokens = 200  # clinical note
    monthly_tokens = avg_note_tokens * total_encounters_per_month
    
    monthly_llm_cost = calculate_llm_cost(monthly_tokens, "gpt-4o-mini")
    yearly_llm_cost = monthly_llm_cost * 12
    
    print(f"   Monthly cost: ${monthly_llm_cost:.2f}")
    print(f"   Yearly cost: ${yearly_llm_cost:.2f}")
    
    print(f"\n📈 SAVINGS WITH EMBEDDINGS:")
    print("-"*70)
    
    yearly_savings = yearly_llm_cost - (initial_embed_cost + monthly_new_embed_cost * 12)
    
    print(f"   First year savings: ${yearly_savings:.2f}")
    print(f"   ROI: {yearly_savings/initial_embed_cost:.0f}x")
    print(f"   Break-even: Immediate (first month)")


def demo_hybrid_approach():
    """
    Demo 3: Hybrid approach - embeddings for search, LLM for refinement
    """
    print("\n\n" + "="*70)
    print("DEMO 3: Hybrid Approach - Best of Both Worlds")
    print("="*70)
    
    print("\n🎯 Strategy:")
    print("   1. Use embeddings to find top 5 similar items (fast & cheap)")
    print("   2. Use LLM to rank/refine only those 5 (accurate)")
    
    # Scenario
    database_size = 50_000
    searches_per_day = 1_000
    days_per_month = 30
    
    avg_item_length = 80
    
    print(f"\n📊 Scenario:")
    print(f"   - Database items: {database_size:,}")
    print(f"   - Searches per day: {searches_per_day:,}")
    print(f"   - Monthly searches: {searches_per_day * days_per_month:,}")
    
    # Approach 1: Pure embeddings
    print(f"\n💰 APPROACH 1: Pure Embeddings")
    print("-"*70)
    
    total_tokens = (database_size * avg_item_length) // 4
    embed_cost = calculate_embedding_cost(total_tokens, "text-embedding-3-small")
    
    print(f"   Setup cost: ${embed_cost:.2f}")
    print(f"   Monthly cost: ${embed_cost:.2f} (one-time)")
    print(f"   Per-search cost: $0.00")
    print(f"   Accuracy: 85% (good)")
    
    # Approach 2: Pure LLM
    print(f"\n💰 APPROACH 2: Pure LLM")
    print("-"*70)
    
    monthly_searches = searches_per_day * days_per_month
    llm_tokens = (avg_item_length * database_size * monthly_searches) // 4
    llm_cost = calculate_llm_cost(llm_tokens, "gpt-4o-mini")
    
    print(f"   Setup cost: $0.00")
    print(f"   Monthly cost: ${llm_cost:.2f}")
    print(f"   Per-search cost: ${llm_cost/monthly_searches:.4f}")
    print(f"   Accuracy: 95% (excellent)")
    
    # Approach 3: Hybrid
    print(f"\n💰 APPROACH 3: Hybrid (Embeddings + LLM)")
    print("-"*70)
    
    # Embeddings for initial search (one-time)
    hybrid_embed_cost = embed_cost
    
    # LLM only for top 5 results per search
    top_k = 5
    hybrid_llm_tokens = (avg_item_length * top_k * monthly_searches) // 4
    hybrid_llm_cost = calculate_llm_cost(hybrid_llm_tokens, "gpt-4o-mini")
    
    hybrid_total = hybrid_embed_cost + hybrid_llm_cost
    
    print(f"   Setup cost: ${hybrid_embed_cost:.2f}")
    print(f"   Monthly LLM cost: ${hybrid_llm_cost:.2f}")
    print(f"   Total monthly: ${hybrid_total:.2f}")
    print(f"   Per-search cost: ${hybrid_llm_cost/monthly_searches:.4f}")
    print(f"   Accuracy: 94% (near-excellent)")
    
    print(f"\n📈 COMPARISON:")
    print("-"*70)
    print(f"   Pure Embeddings: ${embed_cost:.2f}/month - 85% accuracy")
    print(f"   Hybrid: ${hybrid_total:.2f}/month - 94% accuracy (+9%)")
    print(f"   Pure LLM: ${llm_cost:.2f}/month - 95% accuracy (+1%)")
    print(f"\n   💡 Hybrid gives 94% of LLM accuracy at {hybrid_total/llm_cost*100:.1f}% of cost!")


def demo_performance_comparison():
    """
    Demo 4: Performance (speed) comparison
    """
    print("\n\n" + "="*70)
    print("DEMO 4: Performance Comparison - Speed Matters")
    print("="*70)
    
    print("\n⏱️  Average Response Times:")
    print("-"*70)
    
    # Typical response times
    embedding_time = 0.05  # 50ms to get embedding
    local_search_time = 0.001  # 1ms to search 10,000 vectors locally
    llm_time = 1.5  # 1500ms for LLM call
    
    searches_per_session = 10
    
    print(f"   Embedding API call: {embedding_time*1000:.0f}ms")
    print(f"   Local vector search: {local_search_time*1000:.1f}ms (10,000 items)")
    print(f"   LLM API call: {llm_time*1000:.0f}ms")
    
    print(f"\n🔍 Time for {searches_per_session} searches:")
    print("-"*70)
    
    # Embeddings approach: 1 embedding for query + local searches
    embed_approach_time = (embedding_time * searches_per_session) + (local_search_time * searches_per_session)
    
    # LLM approach: 1 LLM call per search
    llm_approach_time = llm_time * searches_per_session
    
    print(f"   Embeddings: {embed_approach_time:.2f}s ({embed_approach_time*1000/searches_per_session:.0f}ms per search)")
    print(f"   LLM: {llm_approach_time:.1f}s ({llm_time*1000:.0f}ms per search)")
    print(f"\n   ⚡ Embeddings are {llm_approach_time/embed_approach_time:.1f}x faster!")
    
    print(f"\n👥 For 1,000 concurrent users:")
    print("-"*70)
    
    users = 1_000
    embed_total = embed_approach_time * users / 60  # minutes
    llm_total = llm_approach_time * users / 60  # minutes
    
    print(f"   Embeddings: {embed_total:.1f} minutes total processing")
    print(f"   LLM: {llm_total:.1f} minutes total processing")
    print(f"\n   💡 Embeddings can handle {llm_total/embed_total:.0f}x more load!")


def demo_practical_recommendations():
    """
    Demo 5: Practical recommendations for different scenarios
    """
    print("\n\n" + "="*70)
    print("DEMO 5: When to Use Each Approach")
    print("="*70)
    
    scenarios = [
        {
            "name": "Drug Interaction Database Search",
            "database_size": 10_000,
            "searches_per_day": 5_000,
            "accuracy_requirement": "High (85%+)",
            "speed_requirement": "Fast (<100ms)",
            "recommendation": "✅ EMBEDDINGS",
            "reason": "High search volume, speed critical, 85% accuracy sufficient"
        },
        {
            "name": "Medical Literature Q&A",
            "database_size": 100_000,
            "searches_per_day": 500,
            "accuracy_requirement": "Very High (95%+)",
            "speed_requirement": "Moderate (1-2s ok)",
            "recommendation": "🔄 HYBRID",
            "reason": "Large database, high accuracy needed, moderate volume"
        },
        {
            "name": "Clinical Decision Support",
            "database_size": 1_000,
            "searches_per_day": 100,
            "accuracy_requirement": "Critical (98%+)",
            "speed_requirement": "Moderate (1-2s ok)",
            "recommendation": "🤖 LLM",
            "reason": "Small database, critical accuracy, low volume, need reasoning"
        },
        {
            "name": "ICD-10 Code Suggestion",
            "database_size": 70_000,
            "searches_per_day": 10_000,
            "accuracy_requirement": "High (90%+)",
            "speed_requirement": "Fast (<200ms)",
            "recommendation": "🔄 HYBRID",
            "reason": "Large database, high volume, need accuracy + speed"
        },
        {
            "name": "Patient Education Material Search",
            "database_size": 5_000,
            "searches_per_day": 2_000,
            "accuracy_requirement": "Good (80%+)",
            "speed_requirement": "Fast (<100ms)",
            "recommendation": "✅ EMBEDDINGS",
            "reason": "High volume, speed important, accuracy requirement reasonable"
        },
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*70}")
        print(f"Scenario {i}: {scenario['name']}")
        print(f"{'='*70}")
        print(f"   Database size: {scenario['database_size']:,} items")
        print(f"   Search volume: {scenario['searches_per_day']:,}/day")
        print(f"   Accuracy needed: {scenario['accuracy_requirement']}")
        print(f"   Speed needed: {scenario['speed_requirement']}")
        print(f"\n   🎯 RECOMMENDATION: {scenario['recommendation']}")
        print(f"   💡 Why: {scenario['reason']}")


def main():
    """
    Run all demonstrations
    """
    print("\n💰 Cost Comparison: Embeddings vs Full LLM")
    print("="*70)
    print("Understanding when to use embeddings vs LLM")
    
    print("\n\nChoose a demo:")
    print("1. Basic cost comparison (10,000 medical terms)")
    print("2. Real use case (ICD-10 code suggestion system)")
    print("3. Hybrid approach (embeddings + LLM)")
    print("4. Performance comparison (speed)")
    print("5. Practical recommendations")
    print("6. Run all demos")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == "1":
        demo_basic_cost_comparison()
    elif choice == "2":
        demo_real_use_case()
    elif choice == "3":
        demo_hybrid_approach()
    elif choice == "4":
        demo_performance_comparison()
    elif choice == "5":
        demo_practical_recommendations()
    elif choice == "6":
        demo_basic_cost_comparison()
        demo_real_use_case()
        demo_hybrid_approach()
        demo_performance_comparison()
        demo_practical_recommendations()
    else:
        print("❌ Invalid choice")
    
    print("\n\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
📊 EMBEDDINGS:
   ✅ One-time cost to create embeddings
   ✅ FREE similarity searches (local computation)
   ✅ Very fast (1-50ms per search)
   ✅ Scales well with high search volume
   ✅ Good accuracy (80-90%)
   💰 Cost: $0.02 per 1M tokens (text-embedding-3-small)

🤖 LLM:
   ✅ Highest accuracy (95%+)
   ✅ Can provide reasoning/explanations
   ✅ Handles complex queries
   ❌ Cost per query (ongoing)
   ❌ Slower (1-3 seconds)
   ❌ Expensive at scale
   💰 Cost: $0.15-$10+ per 1M tokens

🔄 HYBRID (RECOMMENDED):
   ✅ Best of both worlds
   ✅ 90-95% accuracy
   ✅ Cost-effective
   ✅ Fast initial search, accurate refinement
   💰 Cost: Embeddings + LLM on top-K only

DECISION MATRIX:
┌─────────────────────┬─────────────┬─────────────┬─────────────┐
│ Requirement         │ Embeddings  │ Hybrid      │ Pure LLM    │
├─────────────────────┼─────────────┼─────────────┼─────────────┤
│ High search volume  │ ✅ Best     │ ✅ Good     │ ❌ Expensive│
│ Critical accuracy   │ ❌ Limited  │ ✅ Good     │ ✅ Best     │
│ Fast response       │ ✅ Best     │ ✅ Good     │ ❌ Slower   │
│ Large database      │ ✅ Best     │ ✅ Good     │ ❌ Expensive│
│ Need reasoning      │ ❌ No       │ ✅ Yes      │ ✅ Yes      │
│ Low budget          │ ✅ Best     │ ✅ Good     │ ❌ High cost│
└─────────────────────┴─────────────┴─────────────┴─────────────┘

RECOMMENDATION FOR HEALTHCARE:
- Drug interactions → Embeddings (high volume, speed critical)
- ICD-10 codes → Hybrid (large DB, need accuracy)
- Clinical decisions → LLM (need reasoning, critical accuracy)
- Patient education → Embeddings (high volume, good enough accuracy)
""")


if __name__ == "__main__":
    main()
