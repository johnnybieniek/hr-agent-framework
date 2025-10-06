#!/bin/bash

# Setup script for Hugging Face environment on GPU cluster
# Run this script on your university's GPU cluster

echo "Setting up Hugging Face environment for HR Agent Framework..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv-hf
source venv-hf/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements-hf.txt

# Install additional dependencies for GPU clusters
echo "Installing additional GPU dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install bitsandbytes optimum

echo "Setup complete!"

echo "To use this environment:"
echo "Activate the virtual environment: source venv-hf/bin/activate"


