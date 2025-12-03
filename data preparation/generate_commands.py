import os
import random
from openai import OpenAI
import rouge
from tqdm import tqdm

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "***"

# Initialize OpenAI client
client = OpenAI()

# Function to read instructions from a file
def read_instructions(file_path):
    with open(file_path, "r") as file:
        instructions = [line.strip() for line in file if line.strip()]
    return instructions

# Function to write instructions to a file
def write_instructions(file_path, instructions):
    with open(file_path, "w") as file:
        for instruction in instructions:
            file.write(instruction + "\n")

# Function to compute ROUGE-L similarity
def rouge_l_similarity(reference, hypothesis):
    scorer = rouge.Rouge(metrics=["rouge-l"])
    scores = scorer.get_scores(hypothesis, reference)
    return scores[0]["rouge-l"]["f"]

# Function to generate new instructions
def generate_instructions(seed_examples, pool_examples):
    examples = seed_examples + pool_examples
    user_request = " ".join(examples)
    
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": "You are a simulator of the admins of a smart home, please come up with one more new commands to the smart home system. Make sure your commands are: 1. abstract and express user's needs, do not refer to any smart device. 2. in only one sentence concisely and do not give me the sequence number together with the text. 3. give your answer directly like: make the home cozy.",
            },
            {
                "role": "user",
                "content": user_request,
            },
        ],
    )
    
    new_instruction = completion.choices[0].message.content.strip()
    return new_instruction

# Main workflow
def main_workflow(seed_file_path, pool_file_path, max_generated_instructions=5000):
    seed_instructions = read_instructions(seed_file_path)
    instruction_pool = read_instructions(pool_file_path)
    
    generated_count = 0
    
    with tqdm(total=max_generated_instructions, desc="Generating instructions") as pbar:
        while generated_count < max_generated_instructions:
            # Randomly sample 6 instructions from the seed set
            seed_examples = random.sample(seed_instructions, 6)
            
            # Randomly sample 2 instructions from the instruction pool
            if len(instruction_pool) >= 2:
                pool_examples = random.sample(instruction_pool, 2)
            else:
                pool_examples = []
            
            # Generate new instructions
            new_instruction = generate_instructions(seed_examples, pool_examples)
            
            # Check ROUGE-L similarity with existing instructions
            if all(rouge_l_similarity(existing, new_instruction) < 0.7 for existing in seed_instructions + instruction_pool):
                # Add new instruction to the pool
                instruction_pool.append(new_instruction)
                generated_count += 1
                pbar.update(1)  # Update progress bar
            else:
                pass  # Discarded instruction
            
            # Write the updated instruction pool to the file periodically to avoid data loss
            if generated_count % 10 == 0:
                write_instructions(pool_file_path, instruction_pool)
    
    # Final write to ensure all instructions are saved
    write_instructions(pool_file_path, instruction_pool)

# Specify the paths to the seed instructions and pool files
seed_file_path = "/home/iot/Documents/Xinyu/SmartHome/data/seed_instructions.txt"
pool_file_path = "/home/iot/Documents/Xinyu/SmartHome/data/instruction_pool.txt"

# Run the main workflow
main_workflow(seed_file_path, pool_file_path)
