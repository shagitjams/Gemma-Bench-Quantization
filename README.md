Gemma-Bench: Quantization vs. Logical Fidelity

🔬 Project Overview

This research project investigates the trade-offs between computational efficiency and logical reasoning in Large Language Models (LLMs). Using Google’s Gemma-2b-it, I analyzed how different quantization precisions (Float16, 8-bit, and 4-bit) impact performance in high-stakes domains including Mathematics, Python Programming, and Security by Design.

📊 Key Research Findings

-VRAM Efficiency: 4-bit quantization reduced memory consumption by 56% (from 4.9GB to 2.1GB), enabling deployment on consumer-grade hardware.

-Inference Throughput: Achieved an 85% increase in speed, moving from 42.4 to 78.4 tokens per second via 4-bit NF4 optimization.

-Logical Fidelity: Despite aggressive compression, the model retained high reasoning capabilities with only a 2.3% drop in generation confidence (measured via mean transition log-probabilities).

-Research Insight: The data suggests that for Gemma-2b, 4-bit quantization provides a massive boost in speed and memory savings with negligible impact on complex problem-solving.

🛠️ Technical Stack

-Model: google/gemma-2b-it (Instruction Tuned)

-Optimization: bitsandbytes (NF4, Double Quantization)

-Libraries: transformers, accelerate, torch, matplotlib

-Environment: Google Colab (Tesla T4 GPU)

🚀 How to Reproduce

-Request access to Gemma on Hugging Face and generate an Access Token.

-Store your token in Google Colab "Secrets" as HF_TOKEN.

-Run benchmark.py to execute the 10-prompt evaluation suite and generate the performance charts.
