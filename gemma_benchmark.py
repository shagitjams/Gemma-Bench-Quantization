"""
Gemma-Bench: Quantization vs. Logical Fidelity
Description: 
    Benchmarks Google's Gemma-2b-it model across Float16 (Baseline), 8-bit, and 4-bit precisions.
    Designed for Google Colab (T4 GPU). Evaluates reasoning prompts across Math, Python edge-cases, 
    and Security by Design principles.
    
    Metrics Tracked:
    1. VRAM Consumption (MB)
    2. Inference Latency (Tokens/sec)
    3. Generation Confidence (Proxy for Fidelity/Perplexity based on log-probabilities)
"""

import gc
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# ---------------------------------------------------------------------------
# Configuration & Assessment Suite
# ---------------------------------------------------------------------------
MODEL_NAME = "google/gemma-2b-it"
MAX_NEW_TOKENS = 256

# The Evaluation Suite: 10 Complex Reasoning Prompts
PROMPTS = [
    # --- Math Logic ---
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Explain your reasoning step-by-step.",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? Explain your logic.",
    "Solve the following system of equations and explain the substitution process: 3x + 4y = 10 and 2x - y = 3.",
    
    # --- Python Edge-Cases ---
    "Explain the difference between deep copy and shallow copy in Python. Provide a code example where a shallow copy fails to isolate changes in a nested list.",
    "Why is it dangerous to use a mutable object (like a list or dictionary) as a default argument in a Python function? Show a code example demonstrating the unexpected behavior and how to fix it.",
    "What happens in Python if you try to catch multiple exceptions in a single except block but forget the parentheses (e.g., `except TypeError, ValueError:`)? Explain the syntax rules.",
    "Explain the Global Interpreter Lock (GIL) in Python. How does it affect multi-threading versus multi-processing for CPU-bound tasks?",
    
    # --- Security by Design  ---
    "What is Server-Side Request Forgery (SSRF)? Write a Python snippet demonstrating a vulnerable fetch operation and then show a 'Security by Design' refactored version that mitigates SSRF.",
    "Explain how parameterized queries prevent SQL Injection. Provide a vulnerable Python sqlite3 snippet and its secure counterpart.",
    "Describe the concept of 'Buffer Overflow' at a memory level. Why are memory-safe languages like Rust gaining popularity over C/C++ in security-critical infrastructure?"
]


# ---------------------------------------------------------------------------
# Core Benchmark Functions
# ---------------------------------------------------------------------------

def clear_vram():
    """Aggressively clears VRAM to ensure clean isolation between precision tests."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    time.sleep(2)  # Give CUDA allocator time to settle


def load_model(precision: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads the Gemma model in the specified precision utilizing BitsAndBytes.
    """
    print(f"\n[+] Loading {MODEL_NAME} in {precision} precision...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if precision == "Float16 (Baseline)":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16
        )
    elif precision == "8-bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            quantization_config=quant_config
        )
    elif precision == "4-bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            quantization_config=quant_config
        )
    else:
        raise ValueError(f"Unknown precision: {precision}")
        
    return model, tokenizer


def evaluate_model(model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> Dict[str, float]:
    """
    Evaluates the loaded model against the prompt suite.
    Captures Memory, Tokens/Sec, and Generation Confidence (Proxy for Perplexity/Fidelity).
    """
    total_tokens = 0
    total_latency = 0.0
    confidence_scores = []
    
    for idx, prompt in enumerate(PROMPTS):
        # Apply Gemma's specific chat template
        chat = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        start_time = time.time()
        
        # Generate with scores to capture 'fidelity' through log-probs
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        latency = time.time() - start_time
        
        # Calculate Tokens Generated (excluding input prompt)
        generated_sequences = outputs.sequences[:, inputs["input_ids"].shape[1]:]
        num_tokens = generated_sequences.shape[1]
        
        total_tokens += num_tokens
        total_latency += latency
        
        # Calculate Generation Confidence (Proxy for Logical Fidelity)
        # Using the exponential of the mean transition score (log probs)
        transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
        avg_log_prob = transition_scores[0].mean().item()
        confidence = np.exp(avg_log_prob) * 100 # Converted to a 0-100 scale
        confidence_scores.append(confidence)
        
        if idx == 0:  # Print the first generation as a sanity check
            print(f"    - Sample Output (Math Prompt):\n{tokenizer.decode(generated_sequences[0], skip_special_tokens=True).strip()[:150]}...")

    # Aggregated Metrics
    max_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    avg_tokens_per_sec = total_tokens / total_latency
    avg_confidence = np.mean(confidence_scores)
    
    return {
        "VRAM (MB)": max_memory_mb,
        "Tokens/Sec": avg_tokens_per_sec,
        "Fidelity Score": avg_confidence
    }


# ---------------------------------------------------------------------------
# Visualization & Reporting
# ---------------------------------------------------------------------------

def plot_results(results_map: Dict[str, Dict[str, float]]):
    """Generates a side-by-side bar chart of the benchmark results."""
    precisions = list(results_map.keys())
    
    vram_data = [results_map[p]["VRAM (MB)"] for p in precisions]
    throughput_data = [results_map[p]["Tokens/Sec"] for p in precisions]
    fidelity_data = [results_map[p]["Fidelity Score"] for p in precisions]

    # Theme aesthetics
    plt.style.use('ggplot')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['#4285F4', '#34A853', '#EA4335'] # Google Brand Colors

    # Plot 1: VRAM
    axes[0].bar(precisions, vram_data, color=colors)
    axes[0].set_title('Max VRAM Consumption (MB) ↓', fontweight='bold')
    axes[0].set_ylabel('Megabytes')
    
    # Plot 2: Speed (Tokens/sec)
    axes[1].bar(precisions, throughput_data, color=colors)
    axes[1].set_title('Inference Speed (Tokens/sec) ↑', fontweight='bold')
    axes[1].set_ylabel('Tokens per Second')
    
    # Plot 3: Logical Fidelity
    axes[2].bar(precisions, fidelity_data, color=colors)
    axes[2].set_title('Logical Fidelity (Confidence 0-100) ↑', fontweight='bold')
    axes[2].set_ylabel('Avg Generation Confidence Score')

    for ax in axes:
        ax.set_xticklabels(precisions, rotation=15, ha='right')
      
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', padding=3)

    plt.suptitle('Gemma-Bench: Quantization vs. Logical Fidelity', fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main Execution Flow (Colab Entrypoint)
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("🚀 Gemma-Bench initialization...")
    print("=" * 60)
    
    precisions_to_test = ["Float16 (Baseline)", "8-bit", "4-bit"]
    benchmark_results = {}
    
    for precision in precisions_to_test:
        clear_vram()
        try:
            model, tokenizer = load_model(precision)
            print(f"[*] Running evaluation suite for {precision}...")
            metrics = evaluate_model(model, tokenizer)
            
            print(f"[*] Results for {precision}:")
            for k, v in metrics.items():
                print(f"    -> {k}: {v:.2f}")
                
            benchmark_results[precision] = metrics
            
        except Exception as e:
            print(f"[!] Error benchmarking {precision}: {e}")
            
        finally:
            # Rigorous teardown for next iteration
            if 'model' in locals():
                del model
            if 'tokenizer' in locals():
                del tokenizer
            clear_vram()
            
    if benchmark_results:
        print("\n" + "=" * 60)
        print("📊 Generating Visual Analysis...")
        print("=" * 60)
        plot_results(benchmark_results)

if __name__ == "__main__":
    main()
