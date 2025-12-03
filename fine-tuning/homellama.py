from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig, DataCollatorWithPadding
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb
from huggingface_hub import login as hf_login
import torch.nn.functional as F

# Hugging Face login
hf_token = "***"
hf_login(hf_token)

# Parameters
model_id = "meta-llama/Meta-Llama-3-8B"
dataset_name = "unixyhuang/SmartHome-Device-QA"
new_model = "HomeLlama-8B"
hf_model_repo = "unixyhuang/" + new_model

# Configure tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, add_eos_token=True, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# BitsAndBytes configuration for 8-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_quant_type='fp4',  # Adjust as needed
    bnb_8bit_use_double_quant=True,  # Adjust as needed
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, quantization_config=bnb_config, device_map='auto'
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Load and preprocess dataset
dataset = load_dataset(dataset_name, split="train")

def preprocess_data(examples):
    messages = examples['messages']
    response = examples['response']
    assistant_message = {"role": "assistant", "content": response}
    messages.append(assistant_message)
    
    # Encode the full conversation
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    
    return {
        "input_ids": input_ids,
        "messages": examples['messages'],
        "response": examples['response'],
        "text": text
    }

# Apply preprocessing to the dataset
dataset = dataset.map(preprocess_data, remove_columns=['messages', 'response'])

# Split dataset into training and test sets
train_test_split = dataset.train_test_split(test_size=0.05, seed=1234)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Data collator
data_collator = DataCollatorWithPadding(tokenizer)

# PEFT configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=[
        "up_proj",
        "o_proj",
        "v_proj",
        "gate_proj",
        "q_proj",
        "down_proj",
        "k_proj"
    ]
)

model = get_peft_model(model, peft_config)

# Custom Trainer to compute loss
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.get('logits')
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs['input_ids'][..., 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=tokenizer.pad_token_id)
        return (loss, outputs) if return_outputs else loss

# Training arguments
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=100,
    logging_steps=100,
    learning_rate=3e-5,
    weight_decay=0.01,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="linear",
    report_to="wandb"
)

# Initialize wandb
wandb.init(project="llama3-8b-fine-tuning-V1", name="finetuned-homellama")

# Custom Trainer
trainer = CustomTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator
)

# Start training
trainer.train()

# Save model to Hugging Face Hub
model.push_to_hub(hf_model_repo)
tokenizer.push_to_hub(hf_model_repo)
