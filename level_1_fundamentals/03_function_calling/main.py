"""
Project: Function Calling (Tool Use)
Objective: Learn how LLMs can call your Python functions
Concepts: Tool definitions, function calling, multi-step orchestration

Healthcare Use Case: Drug interaction checker, vital signs calculator
"""

import os
import json
from openai import OpenAI
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================
# Healthcare Functions (Tools)
# ============================================

def check_drug_interaction(drug1: str, drug2: str) -> Dict[str, Any]:
    """
    Check if two drugs have known interactions
    In production, this would query a real drug database
    """
    # Simplified mock data for demonstration
    interactions = {
        ("warfarin", "aspirin"): {
            "severity": "high",
            "effect": "Increased risk of bleeding",
            "recommendation": "Avoid combination or monitor closely"
        },
        ("lisinopril", "ibuprofen"): {
            "severity": "moderate",
            "effect": "May reduce effectiveness of blood pressure medication",
            "recommendation": "Use alternative pain reliever like acetaminophen"
        },
        ("metformin", "alcohol"): {
            "severity": "moderate",
            "effect": "Increased risk of lactic acidosis",
            "recommendation": "Limit alcohol consumption"
        }
    }
    
    key = (drug1.lower(), drug2.lower())
    reverse_key = (drug2.lower(), drug1.lower())
    
    if key in interactions:
        return {"interaction_found": True, **interactions[key]}
    elif reverse_key in interactions:
        return {"interaction_found": True, **interactions[reverse_key]}
    else:
        return {
            "interaction_found": False,
            "message": f"No known significant interaction between {drug1} and {drug2}"
        }


def calculate_bmi(weight_kg: float, height_m: float) -> Dict[str, Any]:
    """
    Calculate Body Mass Index and category
    """
    bmi = weight_kg / (height_m ** 2)
    
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal weight"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    
    return {
        "bmi": round(bmi, 1),
        "category": category,
        "weight_kg": weight_kg,
        "height_m": height_m
    }


def calculate_creatinine_clearance(
    age: int,
    weight_kg: float,
    serum_creatinine: float,
    is_female: bool
) -> Dict[str, Any]:
    """
    Calculate creatinine clearance using Cockcroft-Gault equation
    Used for dosing medications in patients with kidney disease
    
    Formula: CrCl = ((140 - age) × weight) / (72 × SCr) × 0.85 (if female)
    """
    crcl = ((140 - age) * weight_kg) / (72 * serum_creatinine)
    
    if is_female:
        crcl *= 0.85
    
    crcl = round(crcl, 1)
    
    # Interpretation
    if crcl >= 90:
        stage = "Normal kidney function"
    elif crcl >= 60:
        stage = "Mild kidney disease (Stage 2)"
    elif crcl >= 30:
        stage = "Moderate kidney disease (Stage 3)"
    elif crcl >= 15:
        stage = "Severe kidney disease (Stage 4)"
    else:
        stage = "Kidney failure (Stage 5)"
    
    return {
        "creatinine_clearance_ml_min": crcl,
        "kidney_function_stage": stage,
        "requires_dose_adjustment": crcl < 60
    }


# ============================================
# Function Definitions for LLM
# ============================================

# Define the tools/functions available to the LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "check_drug_interaction",
            "description": "Check if two medications have known interactions. Use this when a user asks about taking multiple medications together.",
            "parameters": {
                "type": "object",
                "properties": {
                    "drug1": {
                        "type": "string",
                        "description": "The first medication name"
                    },
                    "drug2": {
                        "type": "string",
                        "description": "The second medication name"
                    }
                },
                "required": ["drug1", "drug2"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_bmi",
            "description": "Calculate Body Mass Index given weight and height",
            "parameters": {
                "type": "object",
                "properties": {
                    "weight_kg": {
                        "type": "number",
                        "description": "Weight in kilograms"
                    },
                    "height_m": {
                        "type": "number",
                        "description": "Height in meters"
                    }
                },
                "required": ["weight_kg", "height_m"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_creatinine_clearance",
            "description": "Calculate creatinine clearance (kidney function) using the Cockcroft-Gault equation. Used to determine if medication dose adjustments are needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "age": {
                        "type": "integer",
                        "description": "Patient age in years"
                    },
                    "weight_kg": {
                        "type": "number",
                        "description": "Patient weight in kilograms"
                    },
                    "serum_creatinine": {
                        "type": "number",
                        "description": "Serum creatinine level in mg/dL"
                    },
                    "is_female": {
                        "type": "boolean",
                        "description": "Whether the patient is female"
                    }
                },
                "required": ["age", "weight_kg", "serum_creatinine", "is_female"]
            }
        }
    }
]

# Map function names to actual Python functions
available_functions = {
    "check_drug_interaction": check_drug_interaction,
    "calculate_bmi": calculate_bmi,
    "calculate_creatinine_clearance": calculate_creatinine_clearance
}


def run_conversation(user_query: str) -> str:
    """
    Run a conversation where the LLM can call functions
    
    This demonstrates the full function calling flow:
    1. User asks a question
    2. LLM decides which function(s) to call
    3. We execute the function(s)
    4. LLM uses the results to answer the user
    """
    print(f"\n{'='*60}")
    print(f"User Query: {user_query}")
    print(f"{'='*60}")
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful medical assistant. Use the available tools to help answer questions accurately. Always explain your reasoning."
        },
        {"role": "user", "content": user_query}
    ]
    
    # First API call - LLM decides if it needs to call functions
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"  # Let the model decide when to use tools
    )
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    
    # If the LLM wants to call functions
    if tool_calls:
        print(f"\n🔧 LLM wants to call {len(tool_calls)} function(s):")
        
        # Add the LLM's response to messages
        messages.append(response_message)
        
        # Execute each function call
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"\n  → {function_name}({json.dumps(function_args, indent=4)})")
            
            # Call the actual Python function
            function_to_call = available_functions[function_name]
            function_response = function_to_call(**function_args)
            
            print(f"  ← Result: {json.dumps(function_response, indent=4)}")
            
            # Add function result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": json.dumps(function_response)
            })
        
        # Second API call - LLM uses function results to answer
        print(f"\n💭 LLM is formulating response with function results...")
        
        second_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        
        final_answer = second_response.choices[0].message.content
        
    else:
        # No function calls needed
        final_answer = response_message.content
    
    print(f"\n📋 Final Answer:")
    print(f"{'-'*60}")
    print(final_answer)
    print(f"{'='*60}")
    
    return final_answer


def multi_step_example():
    """
    Example where LLM needs to call multiple functions
    """
    query = """
    I'm 65 years old, weigh 70kg, height 1.75m, and my serum creatinine is 1.5 mg/dL.
    I'm male. Can you:
    1. Calculate my BMI
    2. Check my kidney function
    3. Tell me if I need dose adjustments for medications
    """
    
    run_conversation(query)


def main():
    """
    Run function calling examples
    """
    print("\n🏥 Level 1.3: Function Calling (Tool Use)\n")
    
    # Example 1: Single function call
    print("\n" + "="*60)
    print("EXAMPLE 1: Drug Interaction Check")
    print("="*60)
    run_conversation(
        "Is it safe to take warfarin and aspirin together?"
    )
    
    # Example 2: Different function
    print("\n" + "="*60)
    print("EXAMPLE 2: BMI Calculation")
    print("="*60)
    run_conversation(
        "I weigh 75 kilograms and I'm 1.80 meters tall. What's my BMI?"
    )
    
    # Example 3: Multi-step (uncomment to run)
    # print("\n" + "="*60)
    # print("EXAMPLE 3: Multiple Function Calls")
    # print("="*60)
    # multi_step_example()
    
    # Example 4: No function needed
    print("\n" + "="*60)
    print("EXAMPLE 4: Question Not Requiring Tools")
    print("="*60)
    run_conversation(
        "What are the general symptoms of diabetes?"
    )


if __name__ == "__main__":
    main()
