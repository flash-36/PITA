#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "🚀 Starting Pilot Run to verify setup..."

# Run the pilot check config
python run_parallel.py --config-name=pilot_run

echo "✅ Pilot run finished successfully! You are ready to run the full experiment."
