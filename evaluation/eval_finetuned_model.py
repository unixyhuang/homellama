import random
import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_metric
from transformers import Trainer, TrainingArguments, BitsAndBytesConfig, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login

# Login to Hugging Face
login("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")

# Load the dataset
dataset = datasets.load_dataset("unixyhuang/SmartHome-Device-QA")

# Randomly sample 100 examples
sampled_dataset = dataset["train"].shuffle(seed=42).select(range(100))

# Load the model and tokenizer
model_name = "unixyhuang/HomeLlama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, add_eos_token=True, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# BitsAndBytes configuration for 8-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_quant_type='fp16',  # Adjust as needed
    bnb_8bit_use_double_quant=True,  # Adjust as needed
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, quantization_config=bnb_config, device_map='auto'
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Prepare evaluation metrics
ppl_metric = load_metric("perplexity")
bleu_metric = load_metric("bleu")
rouge_metric = load_metric("rouge")

# Evaluation function
def evaluate_model(sampled_dataset, model, tokenizer):
    references = []
    predictions = []
    ppl_scores = []

    for example in sampled_dataset:
        # Ensure we extract 'system' and 'user' contents properly
        messages = example['messages']
        system_content = next((msg['content'] for msg in messages if msg['role'] == 'system'), "")
        user_content = next((msg['content'] for msg in messages if msg['role'] == 'user'), "")
        input_text = system_content + "\n" + user_content
        
        reference_text = example['response']
        
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=100)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        references.append([reference_text.split()])
        predictions.append(generated_text.split())
        
        # Calculate PPL
        with torch.no_grad():
            loss = model(**inputs, labels=inputs["input_ids"]).loss
        ppl_scores.append(torch.exp(loss).item())

    # Calculate metrics
    bleu_score = bleu_metric.compute(predictions=predictions, references=references)
    rouge_score = rouge_metric.compute(predictions=[" ".join(pred) for pred in predictions], references=[" ".join(ref[0]) for ref in references])
    avg_ppl = sum(ppl_scores) / len(ppl_scores)

    return bleu_score, rouge_score, avg_ppl

# Evaluate the model
bleu_score, rouge_score, avg_ppl = evaluate_model(sampled_dataset, model, tokenizer)

print(f"Original_BLEU Score: {bleu_score['bleu']}")
print(f"Original_ROUGE-L Score: {rouge_score['rougeL']}")
print(f"Original_Average Perplexity: {avg_ppl}")