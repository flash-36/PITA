#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "🚀 Starting combined experiment run..."

# 1. Gemma and Llama
echo "📦 Running Experiment: Gemma and Llama"
python run_parallel.py --config-name=gemma_and_llama

# 2. Phi
echo "📦 Running Experiment: Phi"
python run_parallel.py --config-name=phi

echo "✅ All experiments finished successfully!"