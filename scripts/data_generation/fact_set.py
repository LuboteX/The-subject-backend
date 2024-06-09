import os
import json
import random
import sys
mypath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(mypath)
from utils.strer2 import encoder

def get_random_facts_from_files(folder_path, num_outputs=100):
    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    all_facts = []
    used_files = set()

    while len(all_facts) < num_outputs and len(used_files) < len(files):
        file = random.choice(files)
        if file in used_files:
            continue
        used_files.add(file)
        
        with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Randomly choose an output from the file
        outputs = [item['output']['facts'] for item in data if 'output' in item and 'facts' in item['output']]
        if not outputs:
            continue
        
        chosen_facts = random.choice(outputs)
        all_facts.append(chosen_facts)

        if len(all_facts) >= num_outputs:
            break

    return all_facts

def save_facts_to_json(facts, output_file):
    dataset = {str(i+1): facts[i] for i in range(len(facts))}
    output_data = {"input": {"dataset": dataset}}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    num_iterations=200
    num_outputs=5
    input_folder = 'document/training_data/fact_encoded_data'
    output_folder_raw1 = 'document/training_data/fact_random_raw1_data'
    output_folder_raw2 = 'document/training_data/fact_random_raw2_data'
    file_template_path = "./document/template/fact_quest_template.json"
    for i in range(num_iterations):
        facts = get_random_facts_from_files(input_folder, num_outputs)
        output_file1 = os.path.join(output_folder_raw1, f"random_facts_{i+1:04d}.json")
        output_file2 = os.path.join(output_folder_raw2, f"random_facts_{i+1:04d}.json")
        save_facts_to_json(facts, output_file1)
        encoder.encoder(output_file1,file_template_path,output_file2)
        num_outputs = num_outputs+1
