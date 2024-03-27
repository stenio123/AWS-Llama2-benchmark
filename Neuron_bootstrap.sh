## INSTALL DRIVERS
# Configure Linux for Neuron repository updates
. /etc/os-release
sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
EOF
wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -

# Update OS packages 
sudo apt-get update -y

# Install OS headers 
sudo apt-get install linux-headers-$(uname -r) -y

# Install git 
sudo apt-get install git -y

# install Neuron Driver
sudo apt-get install aws-neuronx-dkms=2.* -y

# Install Neuron Runtime 
sudo apt-get install aws-neuronx-collectives=2.* -y
sudo apt-get install aws-neuronx-runtime-lib=2.* -y

# Install Neuron Tools 
sudo apt-get install aws-neuronx-tools=2.* -y

# Add PATH
export PATH=/opt/aws/neuron/bin:$PATH

## PYTORCH ENV
sudo apt-get update 
sudo apt-get install -y python3-venv

# Create Python venv
python3 -m venv aws_neuron_venv_pytorch 

# Activate Python venv 
source aws_neuron_venv_pytorch/bin/activate 
python -m pip install -U pip 

# Install Jupyter notebook kernel
pip install ipykernel 
python3 -m ipykernel install --user --name aws_neuron_venv_pytorch --display-name "Python (torch-neuronx)"
pip install jupyter notebook
pip install environment_kernels

# Set pip repository pointing to the Neuron repository 
python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

# Install wget, awscli 
python -m pip install wget 
python -m pip install awscli 

# Install Neuron Compiler and Framework
python -m pip install neuronx-cc==2.* torch-neuronx torchvision

## INSTALL TRANSFORMERS neuron-x
source aws_neuron_venv_pytorch/bin/activate
pip install -U transformers_neuronx 
pip install sentencepiece
pip install transformers==4.36.2 

pip install -U "huggingface_hub[cli]"
export PATH="$PATH:~/.local/bin"
huggingface-cli login --token $HUGGINGFACE_TOKEN 
