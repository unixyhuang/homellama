import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import psutil
import random
from transformers import BitsAndBytesConfig
from huggingface_hub import login

# Login to Hugging Face
login("***")

# Function to measure memory usage
def memory_usage_psutil():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 3)  # in GB

# Load the model and tokenizer
model_name = "unixyhuang/HomeLlama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, add_eos_token=True, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# BitsAndBytes configuration for 8-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

# Measure memory usage before loading the model
mem_before_loading = memory_usage_psutil()

# Load model with int8 quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    trust_remote_code=True, 
    quantization_config=bnb_config, 
    device_map='auto'
)

# Measure memory usage after loading the model
mem_after_loading = memory_usage_psutil()
model_loading_memory = mem_after_loading - mem_before_loading

# Load instructions from txt file
with open("/home/iot/Documents/Xinyu/SmartHome/dataset/instrcution_pool.txt", "r") as file:
    instructions = file.readlines()

# Select 100 unique instructions randomly
selected_instructions = random.sample(list(set(instructions)), 100)

# Lists to store memory usage and latency
memory_usages = []
latencies = []

# Test the model
for instruction in selected_instructions:
    inputs = tokenizer(instruction, return_tensors="pt").to(model.device)

    # Measure memory usage before processing
    mem_before = memory_usage_psutil()
    
    # Measure latency
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    end_time = time.time()
    
    # Measure memory usage after processing
    mem_after = memory_usage_psutil()
    
    # Calculate metrics
    memory_usages.append(mem_after - mem_before_loading)  # Total memory used by the model during inference
    latencies.append(end_time - start_time)

# Calculate min, max, mean for memory usage and latency
min_memory_usage = min(memory_usages)
max_memory_usage = max(memory_usages)
mean_memory_usage = sum(memory_usages) / len(memory_usages)

min_latency = min(latencies)
max_latency = max(latencies)
mean_latency = sum(latencies) / len(latencies)

print(f"Model Loading Memory: {model_loading_memory:.2f} GB")
print(f"Memory Usage During Inference - Min: {min_memory_usage:.2f} GB, Max: {max_memory_usage:.2f} GB, Mean: {mean_memory_usage:.2f} GB")
print(f"Latency - Min: {min_latency:.4f} s, Max: {max_latency:.4f} s, Mean: {mean_latency:.4f} s")
