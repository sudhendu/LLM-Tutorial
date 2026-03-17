import csv
import json
import random

def generate_mlx_data(input_csv, output_jsonl):
    instructions = [
        "Where can I find the {} directory?",
        "What is the purpose of the {} folder?",
        "I need to access files related to {}. Where are they?",
        "What kind of content is stored in {}?",
        "Where should I look for {}?",
        "Tell me about the {} directory."
    ]

    dataset = []

    try:
        with open(input_csv, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row['name'].strip() [cite: 1]
                description = row['description'].strip() [cite: 1, 15]

                # Generate 2-3 variations per CSV row to increase dataset size
                for _ in range(2):
                    template = random.choice(instructions)
                    
                    # Randomly decide if we ask about the 'name' or the 'description'
                    if "{}" in template:
                        # Some templates work better with names, some with descriptions
                        if "Where" in template:
                            user_query = template.format(description)
                        else:
                            user_query = template.format(name)
                    
                    assistant_res = f"The {name} directory is used for {description}."
                    
                    # MLX-LM / Llama 3 format
                    formatted_entry = {
                        "text": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{assistant_res}<|eot_id|>"
                    }
                    dataset.append(formatted_entry)

        # Shuffle and save
        random.shuffle(dataset)
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for entry in dataset:
                f.write(json.dumps(entry) + '\n')
                
        print(f"Success! Created {len(dataset)} training examples in {output_jsonl}")

    except FileNotFoundError:
        print("Error: file_directory_descriptions.csv not found.")

if __name__ == "__main__":
    generate_mlx_data('file_directory_descriptions.csv', 'data/train.jsonl')