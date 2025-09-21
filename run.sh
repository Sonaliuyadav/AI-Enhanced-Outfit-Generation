#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Set dataset path
export FASHION_DATASET_PATH="$(pwd)/updated_recommendation.csv"

# Set Flask host and port
export FLASK_APP=main.py
export FLASK_ENV=production
PORT=8080

# Open browser automatically
open "http://localhost:$PORT" &

# Run Flask app
echo "Starting Fashion AI app on http://localhost:$PORT ..."
python main.py

# Deactivate virtual environment after exit
deactivate
echo "Fashion AI app stopped. Goodbye!"
