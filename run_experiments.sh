#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "🚀 Starting combined experiment run..."

# 1. Gemma
echo "📦 Running Experiment: Gemma GSM8K"
python run_parallel.py --config-name=gemma_GSM8K
echo "📦 Running Experiment: Gemma AIME22to24"
python run_parallel.py --config-name=gemma_AIME22to24
echo "📦 Running Experiment: Gemma MATH"
python run_parallel.py --config-name=gemma_MATH
echo "📦 Running Experiment: Gemma IMDBGen"
python run_parallel.py --config-name=gemma_IMDBGen
echo "📦 Running Experiment: Gemma TLDR"
python run_parallel.py --config-name=gemma_TLDR

# 2. Llama-old
echo "📦 Running Experiment: Llama-old GSM8K"
python run_parallel.py --config-name=llama-old_GSM8K
echo "📦 Running Experiment: Llama-old AIME22to24"
python run_parallel.py --config-name=llama-old_AIME22to24
echo "📦 Running Experiment: Llama-old MATH"
python run_parallel.py --config-name=llama-old_MATH
echo "📦 Running Experiment: Llama-old IMDBGen"
python run_parallel.py --config-name=llama-old_IMDBGen
echo "📦 Running Experiment: Llama-old TLDR"
python run_parallel.py --config-name=llama-old_TLDR

# 3. Qwen
echo "📦 Running Experiment: Qwen GSM8K"
python run_parallel.py --config-name=qwen_GSM8K
echo "📦 Running Experiment: Qwen AIME22to24"
python run_parallel.py --config-name=qwen_AIME22to24
echo "📦 Running Experiment: Qwen MATH"
python run_parallel.py --config-name=qwen_MATH
echo "📦 Running Experiment: Qwen IMDBGen"
python run_parallel.py --config-name=qwen_IMDBGen
echo "📦 Running Experiment: Qwen TLDR"
python run_parallel.py --config-name=qwen_TLDR

# 4. Phi
echo "📦 Running Experiment: Phi GSM8K"
python run_parallel.py --config-name=phi_GSM8K
echo "📦 Running Experiment: Phi AIME22to24"
python run_parallel.py --config-name=phi_AIME22to24
echo "📦 Running Experiment: Phi MATH"
python run_parallel.py --config-name=phi_MATH
echo "📦 Running Experiment: Phi IMDBGen"
python run_parallel.py --config-name=phi_IMDBGen
echo "📦 Running Experiment: Phi TLDR"
python run_parallel.py --config-name=phi_TLDR

# 5. Qwen-math
echo "📦 Running Experiment: Qwen-math GSM8K"
python run_parallel.py --config-name=qwen-math_GSM8K
echo "📦 Running Experiment: Qwen-math AIME22to24"
python run_parallel.py --config-name=qwen-math_AIME22to24
echo "📦 Running Experiment: Qwen-math MATH"
python run_parallel.py --config-name=qwen-math_MATH
echo "📦 Running Experiment: Qwen-math IMDBGen"
python run_parallel.py --config-name=qwen-math_IMDBGen
echo "📦 Running Experiment: Qwen-math TLDR"
python run_parallel.py --config-name=qwen-math_TLDR

# 6. Llama
echo "📦 Running Experiment: Llama GSM8K"
python run_parallel.py --config-name=llama_GSM8K
echo "📦 Running Experiment: Llama AIME22to24"
python run_parallel.py --config-name=llama_AIME22to24
echo "📦 Running Experiment: Llama MATH"
python run_parallel.py --config-name=llama_MATH
echo "📦 Running Experiment: Llama IMDBGen"
python run_parallel.py --config-name=llama_IMDBGen
echo "📦 Running Experiment: Llama TLDR"
python run_parallel.py --config-name=llama_TLDR

echo "✅ All experiments finished successfully!"
