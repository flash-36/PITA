#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "🚀 Starting combined experiment run..."

# 1. Gemma
echo "📦 Running Experiment: Gemma GSM8K"
python run_parallel.py --config-name=gemma_GSM8K
echo "📦 Running Experiment: Gemma IMDBGen"
python run_parallel.py --config-name=gemma_IMDBGen
echo "📦 Running Experiment: Gemma TLDR"
python run_parallel.py --config-name=gemma_TLDR

# 2. Llama
echo "📦 Running Experiment: Llama GSM8K"
python run_parallel.py --config-name=llama_GSM8K
echo "📦 Running Experiment: Llama IMDBGen"
python run_parallel.py --config-name=llama_IMDBGen
echo "📦 Running Experiment: Llama TLDR"
python run_parallel.py --config-name=llama_TLDR

# 3. Phi
echo "📦 Running Experiment: Phi GSM8K"
python run_parallel.py --config-name=phi_GSM8K
echo "📦 Running Experiment: Phi IMDBGen"
python run_parallel.py --config-name=phi_IMDBGen
echo "📦 Running Experiment: Phi TLDR"
python run_parallel.py --config-name=phi_TLDR

echo "✅ All experiments finished successfully!"