from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

# Load the model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the input text and move inputs to the same device as the model
inputs = tokenizer("What is deep learning?", return_tensors="pt").to("cuda")

# Ensure the model is on the GPU
model.to("cuda")

# Perform inference 
start_time = time.time() 
outputs = model.generate(**inputs, max_new_tokens=1224, do_sample=True, temperature=0.9, top_k=50, top_p=0.9) 
end_time = time.time() 

# Calculate the execution time 
execution_time = end_time - start_time 
# Calculate the number of generated tokens
# Note: Subtracting the size of input_ids from outputs to get the number of new tokens generated 
generated_tokens = outputs.size(1) - inputs['input_ids'].size(1) 
# Calculate throughput (tokens/sec) 
throughput = generated_tokens / execution_time 
# Decode the generated token IDs back to text 
decoded_sequences = tokenizer.batch_decode(outputs, skip_special_tokens=True) 
# Print results
print(f"Execution time: {execution_time} seconds") 
print(f'Throughput (tokens/sec): {throughput}') 
print(f"Generated sequences: {decoded_sequences}") 
print(f"Input tokens: {inputs['input_ids'].size(1)}, Output tokens: {outputs.size(1)}, Token length: {generated_tokens}")