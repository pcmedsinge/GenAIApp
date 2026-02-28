"""
Exercise 4: Deployment Guide
==============================
Build a deployment script: convert model to Ollama format, create
Modelfile, test endpoints, benchmark performance, create API wrapper.

Learning Objectives:
- Convert fine-tuned models for local deployment
- Create Ollama Modelfiles with medical system prompts
- Build REST API wrappers for model serving
- Benchmark inference performance
- Understand HIPAA-compliant deployment patterns

Run:
    python exercise_4_deployment_guide.py
"""

import json
import os
import time
import subprocess
import random
from collections import defaultdict


# --- Deployment configuration ---
DEFAULT_CONFIG = {
    "model_name": "icd10-medical-coder",
    "base_model_gguf": "icd10-coder.Q4_K_M.gguf",
    "system_prompt": (
        "You are a medical coding assistant specialized in ICD-10 coding. "
        "Given a clinical note, identify the most appropriate ICD-10 diagnosis code. "
        "Always output in the format: ICD-10: [CODE] - [DESCRIPTION]. "
        "Be specific — avoid unspecified codes when clinical detail supports a specific code."
    ),
    "temperature": 0.1,
    "num_ctx": 4096,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
}


def generate_modelfile(config: dict) -> str:
    """Generate an Ollama Modelfile."""

    modelfile = f"""FROM ./{config['base_model_gguf']}

SYSTEM \"\"\"{config['system_prompt']}\"\"\"

PARAMETER temperature {config['temperature']}
PARAMETER num_ctx {config['num_ctx']}
PARAMETER top_p {config['top_p']}
PARAMETER repeat_penalty {config['repeat_penalty']}

TEMPLATE \"\"\"{{{{ if .System }}}}{{{{ .System }}}}{{{{ end }}}}
{{{{ if .Prompt }}}}Clinical Note: {{{{ .Prompt }}}}{{{{ end }}}}
{{{{ .Response }}}}\"\"\"
"""
    return modelfile


def generate_conversion_script(config: dict) -> str:
    """Generate the model conversion script."""

    script = f"""#!/bin/bash
# ICD-10 Medical Coder — Model Conversion Script
# Converts fine-tuned HF model to GGUF format for Ollama deployment

set -e

MERGED_MODEL_DIR="./icd10-model-merged"
OUTPUT_GGUF="./{config['base_model_gguf']}"
MODEL_NAME="{config['model_name']}"

echo "=== Step 1: Merge LoRA Weights ==="
python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print('Loading base model...')
base = AutoModelForCausalLM.from_pretrained(
    'mistralai/Mistral-7B-v0.1',
    torch_dtype=torch.float16,
    device_map='auto',
)

print('Loading LoRA adapter...')
model = PeftModel.from_pretrained(base, './icd10-lora-adapter')

print('Merging weights...')
merged = model.merge_and_unload()

print('Saving merged model...')
merged.save_pretrained('$MERGED_MODEL_DIR')
AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1').save_pretrained('$MERGED_MODEL_DIR')
print('Done!')
"

echo "=== Step 2: Convert to GGUF ==="
# Clone llama.cpp if not present
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp && pip install -r requirements.txt && cd ..
fi

python llama.cpp/convert_hf_to_gguf.py \\
    "$MERGED_MODEL_DIR" \\
    --outfile "$OUTPUT_GGUF" \\
    --outtype q4_K_M

echo "=== Step 3: Create Modelfile ==="
cat > Modelfile << 'MODELFILE_EOF'
{generate_modelfile(config).strip()}
MODELFILE_EOF

echo "=== Step 4: Import into Ollama ==="
ollama create "$MODEL_NAME" -f Modelfile

echo "=== Step 5: Verify ==="
ollama list | grep "$MODEL_NAME"

echo "=== Step 6: Test ==="
ollama run "$MODEL_NAME" "67-year-old male with chest pain, ST elevation, elevated troponin."

echo ""
echo "Deployment complete! Model available as: $MODEL_NAME"
"""
    return script


def generate_api_wrapper() -> str:
    """Generate a Python API wrapper for the deployed model."""

    wrapper = '''"""
ICD-10 Medical Coder — API Wrapper
Provides a clean Python interface to the locally deployed Ollama model.
"""

import requests
import json
import time
from typing import Optional


class ICD10Coder:
    """Client for the locally deployed ICD-10 coding model."""

    def __init__(self, model_name: str = "icd10-medical-coder",
                 base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

    def code_note(self, clinical_note: str, temperature: float = 0.1,
                  timeout: int = 30) -> dict:
        """Send a clinical note for ICD-10 coding.

        Args:
            clinical_note: The clinical note text to code.
            temperature: Sampling temperature (lower = more deterministic).
            timeout: Request timeout in seconds.

        Returns:
            dict with 'code', 'description', 'raw_response', 'latency_ms'
        """
        start = time.time()

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": clinical_note,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": 100,
                }
            },
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()
        latency = (time.time() - start) * 1000

        raw = result.get("response", "").strip()

        # Parse ICD-10 code from response
        code, description = "", ""
        if "ICD-10:" in raw:
            after = raw.split("ICD-10:")[1].strip()
            parts = after.split(" - ", 1)
            code = parts[0].strip()
            description = parts[1].strip() if len(parts) > 1 else ""

        return {
            "code": code,
            "description": description,
            "raw_response": raw,
            "latency_ms": latency,
            "model": self.model_name,
        }

    def batch_code(self, notes: list, temperature: float = 0.1) -> list:
        """Code multiple clinical notes."""
        results = []
        for note in notes:
            result = self.code_note(note, temperature)
            results.append(result)
        return results

    def health_check(self) -> bool:
        """Check if the Ollama server is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.ConnectionError:
            return False

    def model_info(self) -> Optional[dict]:
        """Get model information from Ollama."""
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": self.model_name},
                timeout=10,
            )
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return None


# Usage example
if __name__ == "__main__":
    coder = ICD10Coder()

    if not coder.health_check():
        print("Ollama server not running. Start with: ollama serve")
        exit(1)

    result = coder.code_note(
        "67-year-old male with acute substernal chest pain radiating to "
        "left arm, diaphoresis. ECG shows ST elevation in V1-V4. "
        "Troponin I elevated at 3.2 ng/mL."
    )

    print(f"ICD-10: {result['code']} - {result['description']}")
    print(f"Latency: {result['latency_ms']:.0f}ms")
'''
    return wrapper


def simulate_benchmark(num_requests: int = 50) -> dict:
    """Simulate performance benchmarks."""

    random.seed(42)

    results = {
        "num_requests": num_requests,
        "latencies_ms": [],
        "tokens_per_sec": [],
        "errors": 0,
    }

    for _ in range(num_requests):
        # Simulate realistic latency distribution
        if random.random() < 0.95:  # 95% normal
            latency = random.gauss(250, 80)  # ~250ms avg
            tps = random.gauss(45, 10)  # ~45 tokens/sec
        else:  # 5% slow (cold start, GC, etc.)
            latency = random.gauss(800, 200)
            tps = random.gauss(15, 5)

        latency = max(50, latency)
        tps = max(5, tps)

        results["latencies_ms"].append(latency)
        results["tokens_per_sec"].append(tps)

    return results


def print_benchmark_report(results: dict):
    """Print benchmark results."""

    latencies = results["latencies_ms"]
    tps = results["tokens_per_sec"]

    sorted_lat = sorted(latencies)
    p50 = sorted_lat[len(sorted_lat) // 2]
    p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
    p99 = sorted_lat[int(len(sorted_lat) * 0.99)]

    print(f"\n  Requests:   {results['num_requests']}")
    print(f"  Errors:     {results['errors']}")
    print(f"\n  Latency (ms):")
    print(f"    Min:  {min(latencies):>8.0f}")
    print(f"    P50:  {p50:>8.0f}")
    print(f"    P95:  {p95:>8.0f}")
    print(f"    P99:  {p99:>8.0f}")
    print(f"    Max:  {max(latencies):>8.0f}")
    print(f"    Avg:  {sum(latencies)/len(latencies):>8.0f}")
    print(f"\n  Tokens/sec:")
    print(f"    Min:  {min(tps):>8.1f}")
    print(f"    Avg:  {sum(tps)/len(tps):>8.1f}")
    print(f"    Max:  {max(tps):>8.1f}")

    # Latency histogram
    print(f"\n  Latency distribution:")
    buckets = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 1000), (1000, 2000)]
    for lo, hi in buckets:
        count = sum(1 for l in latencies if lo <= l < hi)
        bar = "█" * int(count / len(latencies) * 40)
        print(f"    {lo:>5d}-{hi:<5d}ms  {count:3d}  {bar}")


def check_ollama_status():
    """Check if Ollama is installed and running."""

    print("\n--- Checking Ollama Status ---")

    # Check if installed
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"  ✓ Ollama installed: {result.stdout.strip()}")
        else:
            print(f"  ✗ Ollama not found")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"  ✗ Ollama not installed")
        print(f"    Install: curl -fsSL https://ollama.com/install.sh | sh")
        return False

    # Check if running
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            print(f"  ✓ Ollama server running ({len(models)} models loaded)")
            for m in models[:5]:
                name = m.get("name", "unknown")
                size = m.get("size", 0) / 1e9
                print(f"      {name} ({size:.1f} GB)")
            return True
    except Exception:
        pass

    print(f"  ✗ Ollama server not running")
    print(f"    Start with: ollama serve")
    return False


def main():
    """Build deployment guide and scripts."""

    print("=" * 60)
    print("Exercise 4: Deployment Guide")
    print("=" * 60)

    config = DEFAULT_CONFIG

    # --- Step 1: Check environment ---
    print("\n" + "=" * 60)
    print("Step 1: Environment Check")
    print("=" * 60)
    ollama_ready = check_ollama_status()

    # --- Step 2: Generate Modelfile ---
    print("\n" + "=" * 60)
    print("Step 2: Ollama Modelfile")
    print("=" * 60)
    modelfile = generate_modelfile(config)
    print(f"\n  Generated Modelfile:")
    for line in modelfile.strip().split("\n"):
        print(f"    {line}")

    # Save Modelfile
    base_dir = os.path.dirname(__file__) or "."
    modelfile_path = os.path.join(base_dir, "Modelfile")
    with open(modelfile_path, "w") as f:
        f.write(modelfile)
    print(f"\n  Saved to: {modelfile_path}")

    # --- Step 3: Generate conversion script ---
    print("\n" + "=" * 60)
    print("Step 3: Conversion Script")
    print("=" * 60)
    conv_script = generate_conversion_script(config)
    conv_path = os.path.join(base_dir, "deploy.sh")
    with open(conv_path, "w") as f:
        f.write(conv_script)
    os.chmod(conv_path, 0o755)
    print(f"  Generated: {conv_path}")
    print(f"  Run with: bash deploy.sh")

    # --- Step 4: Generate API wrapper ---
    print("\n" + "=" * 60)
    print("Step 4: API Wrapper")
    print("=" * 60)
    api_wrapper = generate_api_wrapper()
    api_path = os.path.join(base_dir, "icd10_api.py")
    with open(api_path, "w") as f:
        f.write(api_wrapper)
    print(f"  Generated: {api_path}")
    print(f"  Usage:")
    print(f"    from icd10_api import ICD10Coder")
    print(f"    coder = ICD10Coder()")
    print(f'    result = coder.code_note("Patient presents with...")')

    # --- Step 5: Benchmark ---
    print("\n" + "=" * 60)
    print("Step 5: Performance Benchmark (Simulated)")
    print("=" * 60)
    benchmark = simulate_benchmark(num_requests=100)
    print_benchmark_report(benchmark)

    # --- Step 6: Deployment checklist ---
    print("\n" + "=" * 60)
    print("Step 6: Deployment Checklist")
    print("=" * 60)
    checklist = [
        ("Install Ollama", "curl -fsSL https://ollama.com/install.sh | sh"),
        ("Fine-tune model", "Complete Level 7 Projects 03-04"),
        ("Merge LoRA weights", "python merge_model.py"),
        ("Convert to GGUF", "python llama.cpp/convert_hf_to_gguf.py ..."),
        ("Create Modelfile", "Generated ✓"),
        ("Import into Ollama", f"ollama create {config['model_name']} -f Modelfile"),
        ("Test model", f"ollama run {config['model_name']} \"test note\""),
        ("Run benchmarks", "python exercise_4_deployment_guide.py"),
        ("Deploy API wrapper", "python icd10_api.py"),
        ("Set up monitoring", "Track latency, accuracy, usage"),
    ]

    print()
    for i, (step, cmd) in enumerate(checklist, 1):
        print(f"  {'☐':2s} {i:2d}. {step:30s} → {cmd}")

    # --- HIPAA compliance notes ---
    print(f"""
{'=' * 60}
HIPAA COMPLIANCE NOTES
{'=' * 60}

  ✓ All inference runs on local hardware
  ✓ No patient data transmitted to external services
  ✓ Model weights stored on encrypted local storage
  ✓ API wrapper runs on localhost only by default
  ✓ No telemetry or logging of PHI to external services

  Additional recommendations:
  • Enable disk encryption on model storage
  • Implement audit logging for all API requests
  • Restrict network access (firewalls, no external ports)
  • Regular model evaluation for accuracy drift
  • BAA not required (no third-party cloud services)

{'=' * 60}
  ✓ Deployment guide complete!
  
  Files generated:
    Modelfile           — Ollama model configuration
    deploy.sh           — Automated deployment script
    icd10_api.py        — Python API wrapper
""")


if __name__ == "__main__":
    main()
