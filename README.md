# AWS Llama 2 Benchmark

## Overview
This code compares the execution time and costs of running Llama 2 7b on the following AWS instances:
- g5.2xlarge
- inf2.xlarge
 

|          | g5.2xlarge | inf2.xlarge |
|----------|------------|-------------|
| Price    | $1.212/hour | $0.758/hour |
| FP16 TFLOPS | 125 | 180 |
| Tokens per second | 21.748 | 36.40 |
| Cost per million tokens * | $15.48 | $5.784 |
| Execution time | 56.2809 seconds | 33.4579 seconds|

*Cost per million tokens: ($price / (X tokens/sec * 3600)) * 1,000,000

## Note on Neuron Architecture
To leverage hardware optimization, we will compile the model for Neuron architecture, a one-time process. Due to Llama 2 7b's memory demands, compile it on an inf2.8xlarge, then run inference on a smaller inf2.xlarge for the enhanced performance documented in the above table.

## Note on Llama 2 7b
The following examples will download the model from HuggingFace, using "meta-llama/Llama-2-7b-hf".

Requirements:
- Create HuggingFace login and API key
- For the meta-llama, you will need to accept the [terms and conditions](https://huggingface.co/meta-llama) of both HuggingFace and Meta. 

## Benchmark

### g5.2xlarge
- Launch an g5.2xlarge instance with AMI "Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.1.0 (Ubuntu 20.04) 20240312", or equivalent latest version
- Export your [HuggingFace API token](https://huggingface.co/docs/hub/en/security-tokens) with 
```
export HUGGINGFACE_TOKEN=YOUR-TOKEN
```
- Run `CUDA_bootstrap.sh`
- Execute `python CUDA_execution.py`

### inf2.xlarge
#### Compile in inf2.8xlarge
- Launch a inf2.8xlarge with AMI "Deep Learning AMI Neuron (Ubuntu 22.04) 20240318", or equivalent latest version. Use 150Gb storage.
- Attach [IAM role](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html) that allows this instance to upload files to S3, and create a S3 bucket. 
- Export your [HuggingFace API token](https://huggingface.co/docs/hub/en/security-tokens) with 
```
export HUGGINGFACE_TOKEN=YOUR-TOKEN
```
- Run `Neuron_bootstrap.sh`
- Go to the python venv by running
```
source aws_neuron_venv_pytorch/bin/activate 
```
- Update 'YOUR-S3-BUCKET-NAME' in Neuron_compile.py and execute `python Neuron_compile.py`
- You will get some warnings and console will halt until execution complete, which should take about 6-12 minutes

#### Run inference in inf2.xlarge
- Launch a inf2.xlarge with AMI "Deep Learning AMI Neuron (Ubuntu 22.04) 20240318", or equivalent latest version
- Attach [IAM role](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html) that allows this instance to read files from S3, 
- Export your [HuggingFace API token](https://huggingface.co/docs/hub/en/security-tokens) with 
```
export HUGGINGFACE_TOKEN=YOUR-TOKEN
```
- Run `Neuron_bootstrap.sh`
- Go to the python venv by running
```
source aws_neuron_venv_pytorch/bin/activate 
```
- Download models by running
```
aws s3 cp s3://YOUR-BUCKET-NAME/Llama-2-7b-split ./Llama-2-7b-split --recursive
aws s3 cp s3://YOUR-BUCKET-NAME/Llama-2-7b-neuron ./Llama-2-7b-neuron --recursive
```
- Execute `python Neuron_execution.py`

## References
[Bootstrap Neuron on Ubuntu](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/neuron-setup/pytorch/neuronx/ubuntu/torch-neuronx-ubuntu20-base-dlami.html#setup-torch-neuronx-ubuntu20-base-dlami)

[Neuron model serialization](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/transformers-neuronx/transformers-neuronx-developer-guide.html?highlight=serialization#serialization-support-beta)

[Inferentia2 architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/inferentia2.html#inferentia2-arch)

