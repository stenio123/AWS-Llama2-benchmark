sudo apt-get update
sudo apt-get upgrade 
sudo apt install python3-pip

pip install transformers torch

pip install -U "huggingface_hub[cli]"
export PATH="$PATH:~/.local/bin"
huggingface-cli login --token $HUGGINGFACE_TOKEN 
