import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import profile
import collaboration

# Load model and tokenizer with FP16 precision
model_name = "unixyhuang/HomeLlama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()

dialog_history = []
user_profile_path = "/home/iot/Documents/Xinyu/SmartHome/homellama/user_profile.txt"

system_prompt_text = "You are an AI that controls a smart home. You receive user commands and then generate action plans which assign changes to devices in response. Directly give your response in one sentence concisely and do not give any other information."

def build_inputs_from_history(dialog_history):
    """
    Construct the input tensor from dialog history for the model.
    """
    messages = system_prompt_text + "\n\n"
    for message in dialog_history:
        if message['role'] == 'user':
            messages += f"User: {message['content']}\n"
        else:
            messages += f"Assistant: {message['content']}\n"
    return messages

def generate_response(user_input):
    global dialog_history
    dialog_history.append({"role": "user", "content": user_input})

    chat_history = build_inputs_from_history(dialog_history)
    inputs = tokenizer(chat_history, return_tensors="pt").to('cuda')
    attention_mask = inputs['input_ids'].ne(tokenizer.pad_token_id).to('cuda')

    with torch.cuda.amp.autocast():
        outputs = model.generate(inputs['input_ids'], attention_mask=attention_mask, max_new_tokens=50, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id)
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    response = response.split('Assistant:')[-1].strip()
    dialog_history.append({"role": "Assistant", "content": response})

    return response

def summarize_and_update_profile():
    summary = profile.summarize_dialog_history(dialog_history)
    profile.append_user_profile(summary, user_profile_path)

def get_response(user_input):
    response = generate_response(user_input)
    print(f"Assistant: {response}")
    return response

def main():
    print("Welcome to Your Smart Home! I am your assistant. Type your commands or type 'exit' to quit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = get_response(user_input)
        
        feedback = input("Are you satisfied with the response? (yes/no/advice): ").strip().lower()
        
        if feedback == "yes":
            summarize_and_update_profile()
        elif feedback == "no":
            api_key = input("Enter your OpenAI API Key: ").strip()
            remote_response = collaboration.get_remote_assistance(api_key, user_input)
            print(f"Assistant (remote): {remote_response}")
        elif feedback == "advice":
            advice = input("Please provide your advice: ").strip()
            dialog_history.append({"role": "Assistant", "content": advice})
            summarize_and_update_profile()

if __name__ == "__main__":
    main()