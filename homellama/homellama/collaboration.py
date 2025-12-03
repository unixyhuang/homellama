import openai
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load local model and tokenizer with FP16 precision
model_name = "unixyhuang/HomeLlama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()

def generate_similar_instructions(user_input):
    prompt = f"Generate 9 instructions similar to the following: {user_input}\n"
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    attention_mask = inputs['input_ids'].ne(tokenizer.pad_token_id).to('cuda')

    with torch.cuda.amp.autocast():
        outputs = model.generate(inputs['input_ids'], attention_mask=attention_mask, max_new_tokens=150, num_return_sequences=9, pad_token_id=tokenizer.eos_token_id)
    
    similar_instructions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return similar_instructions

def get_remote_assistance(api_key, user_input):
    openai.api_key = api_key

    similar_instructions = generate_similar_instructions(user_input)
    all_instructions = [user_input] + similar_instructions

    responses = []
    for instruction in all_instructions:
        response = openai.Completion.create(
            model="gpt-4",
            prompt=instruction,
            max_tokens=150
        )
        responses.append(response.choices[0].text.strip())

    user_response = responses[0]
    return user_response