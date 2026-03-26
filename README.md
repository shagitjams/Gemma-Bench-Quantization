
Gemma-Bench: Quantization vs. Logical Fidelity
🔬 Project Overview
This repository contains a research-focused benchmarking suite designed to evaluate the trade-offs between computational efficiency and logical reasoning in Large Language Models (LLMs). Using Google’s Gemma-2b-it, I analyzed how different quantization precisions (Float16, 8-bit, and 4-bit) impact performance in high-stakes domains like Mathematics, Python Programming, and Security by Design.

📊 Key Research Findings

Metric	            Float16 (Baseline)	    8-bit   	    4-bit (Extreme)
VRAM Usage	           4874.3 MB	      2697.5 MB	     2138.8 MB
Inference Speed   	42.4 Tokens/sec	   72.3 Tokens/sec	 78.4 Tokens/sec
Fidelity Score	        67.0%	          65.5%	             64.7%

Analysis:

Efficiency Gains: 4-bit quantization achieved a 56% reduction in VRAM and an 85% increase in throughput compared to the Float16 baseline.

Minimal Fidelity Loss: Despite the aggressive compression, the Logical Fidelity (measured via mean transition log-probabilities) dropped by only 2.3%, suggesting that Gemma-2b is highly resilient to 4-bit NormalFloat (NF4) quantization.

Research Insight: The 4-bit model is the optimal choice for on-device deployment or low-latency security auditing tools where memory is a primary constraint.

🛠️ Technical Stack
Model: google/gemma-2b-it (Instruction Tuned)

Optimization: bitsandbytes (NF4 Quantization, Double Quantization)

Frameworks: transformers, accelerate, torch

Evaluation: Custom-built suite of 10 complex reasoning prompts focused on Systems Architecture and Security.

🚀 How to Reproduce
Clone the repo: git clone https://github.com/shagitjams/Gemma-Bench-Quantization.git

Install dependencies: pip install -r requirements.txt

Run in Colab: Ensure a T4 GPU is active and your Hugging Face token is stored in "Secrets."

Execute: python benchmark.py

Run benchmarking_script.py in a GPU-enabled environment.

Provide your Hugging Face API token when prompted.
