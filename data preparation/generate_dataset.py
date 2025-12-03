import os
from openai import OpenAI
import json
from tqdm import tqdm

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Initialize OpenAI client
client = OpenAI()

# Load commands
def read_commands(file_path):
    with open(file_path, "r") as file:
        commands = [line.strip() for line in file if line.strip()]
    return commands

# Generate response from GPT
def generate_response(user_content):
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": "You are a smart home assistant. Given user commands, and consider all the off-the-shelf smart home devices, please directly output one subset which contains all possible relevant devices to the command. Please directly generate a concise set like: {lights, thermostat, curtain}. Do not give any other information. ",
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
    )
    return completion.choices[0].message.content.strip()

# save as jsonl file
def save_to_jsonl(data, file_path):
    with open(file_path, 'a') as file:
        for entry in data:
            file.write(json.dumps(entry) + '\n')

# main function
def main(commands_file_path, output_file_path):
    commands = read_commands(commands_file_path)
    data = []
    
    for i, command in enumerate(tqdm(commands, desc="Processing commands")):
        response = generate_response(command)
        entry = {
            "messages": [
                {"role": "system", "content": "You are a smart home assistant. Given user commands, and consider all the off-the-shelf smart home devices, please directly output one subset which contains all possible relevant devices to the command. Please directly generate a concise set like: {lights, thermostat, curtain}. Do not give any other information."},
                {"role": "user", "content": command},
                {"role": "assistant", "content": response}
            ]
        }
        data.append(entry)

        # Save the generated results
        if (i + 1) % 10 == 0:
            save_to_jsonl(data, output_file_path)
            data = []
    
    # Save the left results
    if data:
        save_to_jsonl(data, output_file_path)

# File path
commands_file_path = "/home/iot/Documents/Xinyu/SmartHome/data/instruction_pool.txt"
output_file_path = "/home/iot/Documents/Xinyu/SmartHome/dataset.jsonl"

# Run
main(commands_file_path, output_file_path)