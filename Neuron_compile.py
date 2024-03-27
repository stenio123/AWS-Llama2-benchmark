import os
import time
import torch
from transformers import AutoTokenizer
from transformers_neuronx.llama.model import LlamaForSampling
from transformers_neuronx.config import NeuronConfig, QuantizationConfig
from transformers import LlamaForCausalLM
from transformers_neuronx.module import save_pretrained_split
from s3_utils import upload_folder_to_s3

bucket_name = 'YOUR-BUCKET-NAME'

model_name = 'meta-llama/Llama-2-7b-hf'
model = LlamaForCausalLM.from_pretrained(model_name)

save_pretrained_split(model, './Llama-2-7b-split')
neuron_config = NeuronConfig(
    quant=QuantizationConfig(quant_dtype='s8', dequant_dtype='bf16'),
)
# load meta-llama/Llama-2-13b to the NeuronCores with 24-way tensor parallelism and run compilation
batch_size = 1
neuron_model = LlamaForSampling.from_pretrained('./Llama-2-7b-split', batch_size=batch_size, tp_degree=2, amp='bf16', n_positions=2048, neuron_config=neuron_config)
neuron_model.to_neuron()
# Save the compiled Neuron model
neuron_model.save('./Llama-2-7b-neuron')

upload_folder_to_s3('Llama-2-7b-split', bucket_name, 'Llama-2-7b-split')
upload_folder_to_s3('Llama-2-7b-neuron', bucket_name, 'Llama-2-7b-neuron')