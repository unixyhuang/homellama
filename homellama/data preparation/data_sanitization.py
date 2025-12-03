import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_instructions(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        instructions = file.readlines()
    instructions = [instruction.strip() for instruction in instructions if instruction.strip()]
    return instructions

def save_instructions(file_path, instructions):
    with open(file_path, 'w', encoding='utf-8') as file:
        for instruction in instructions:
            file.write(instruction + '\n')

def remove_similar_instructions(instructions, similarity_threshold=0.8):
    vectorizer = TfidfVectorizer().fit_transform(instructions)
    similarity_matrix = cosine_similarity(vectorizer)

    to_remove = set()
    for i in range(len(instructions)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(instructions)):
            if similarity_matrix[i, j] > similarity_threshold:
                to_remove.add(j)

    filtered_instructions = [instruction for idx, instruction in enumerate(instructions) if idx not in to_remove]
    return filtered_instructions

def filter_instructions(instructions, max_words=20):
    filtered_instructions = [instruction for instruction in instructions if len(instruction.split()) <= max_words]
    return filtered_instructions

def main():
    input_file = '/home/iot/Documents/Xinyu/SmartHome/data/filtered_instruction_pool.txt'
    output_file = '/home/iot/Documents/Xinyu/SmartHome/data/final_instrcution_pool.txt'
    similarity_threshold = 0.7

    instructions = load_instructions(input_file)
    instructions = filter_instructions(instructions)
    filtered_instructions = remove_similar_instructions(instructions, similarity_threshold)
    save_instructions(output_file, filtered_instructions)
    print(f'Filtered instructions saved to {output_file}')

if __name__ == '__main__':
    main()


