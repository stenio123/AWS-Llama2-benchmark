import os
import time
import torch
from transformers import AutoTokenizer
from transformers_neuronx.llama.model import LlamaForSampling
from transformers_neuronx.config import NeuronConfig, QuantizationConfig
from transformers import LlamaForCausalLM
from s3_utils import download_folder_from_s3

bucket_name = 'YOUR-BUCKET-NAME'
model_name = 'meta-llama/Llama-2-7b-hf'

#download_folder_from_s3(bucket_name, 'Llama-2-7b-split', 'Llama-2-7b-split')
#download_folder_from_s3(bucket_name, 'Llama-2-7b-neuron', 'Llama-2-7b-neuron')

# Load the Neuron model
neuron_config = NeuronConfig(
    quant=QuantizationConfig(quant_dtype='s8', dequant_dtype='bf16'),
)
# load meta-llama/Llama-2-13b to the NeuronCores with 24-way tensor parallelism and run compilation
batch_size = 1
neuron_model = LlamaForSampling.from_pretrained('Llama-2-7b-split', batch_size=batch_size, tp_degree=2, amp='bf16', n_positions=2048, neuron_config=neuron_config)
neuron_model.load('Llama-2-7b-neuron') # Load the compiled Neuron artifacts
neuron_model.to_neuron() # Load the model weights but skip compilation

# construct a tokenizer and encode prompt text
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "What is deep learning?"

tokenizer.pad_token = tokenizer.eos_token
#input_ids = torch.as_tensor([tokenizer.encode(text) for text in batch_prompts])

# run inference with top-k sampling
with torch.inference_mode():
    # input_ids = torch.as_tensor([tokenizer.encode(text) for text in batch_prompts])
    start = time.time()
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated_sequences = neuron_model.sample(input_ids, sequence_length=1224, top_k=1)
    elapsed = time.time() - start
    output_sequences = [tokenizer.decode(seq) for seq in generated_sequences]
    print(f'generated sequences {output_sequences} in {elapsed} seconds')
    print(f'Throughput (tokens/sec): {(generated_sequences.size()[1]-input_ids.size()[1])/elapsed}')
    print("input tokens:", input_ids.size()[1], "output tokens: ", generated_sequences.size(), "token length:", generated_sequences.size())

