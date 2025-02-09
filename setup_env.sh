#!/bin/bash

echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

echo "Installing required system packages..."
sudo apt install python3 python3-venv python3-pip build-essential ffmpeg -y

TEMP_DIR="/tmp/rudn3_clone"
TARGET_DIR="/assistant"

echo "Copying files  to $TARGET_DIR"
sudo mkdir -p "$TARGET_DIR"
sudo cp -r "$TEMP_DIR/appServer/"* "$TARGET_DIR"

echo "Creating virtual environment..."
python3 -m venv my_env
source my_env/bin/activate

echo "Installing Python packages..."
pip install --upgrade pip
pip install gigachat pyannote.audio accelerate fastapi[all] ffmpeg-python aiofiles pymongo uvicorn python-multipart

echo "Verifying CUDA availability..."
pip install torch torchvision torchaudio transformers
python -c "import torch; print(torch.cuda.is_available())"

echo "Installation complete!"
