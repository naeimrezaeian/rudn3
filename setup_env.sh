#!/bin/bash

echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

echo "Installing required system packages..."
sudo apt install python3 python3-venv python3-pip build-essential ffmpeg -y

echo "Creating virtual environment..."
python3 -m venv my_env
source my_env/bin/activate

echo "Installing Python packages..."
pip install --upgrade pip
pip install gigachat pyannote.audio 'accelerate>=0.26.0' fastapi[all] ffmpeg-python aiofiles pymongo uvicorn python-multipart
pip install --upgrade git+https://github.com/huggingface/transformers.git

echo "Verifying CUDA availability..."
pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.is_available())"

echo "Installation complete!"
