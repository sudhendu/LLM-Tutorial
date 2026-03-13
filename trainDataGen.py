import json
import random

# Seed topics to ensure diversity
topics = ["Python debugging", "Healthy meal prep", "Local LLM setup", "Time management", "Quantum physics"]
styles = ["helpful and professional", "witty and concise", "educational and detailed"]

def generate_sample(topic, style):
    # In a real scenario, you would call an LLM API here.
    # For now, this is the structure MLX-LM expects:
    instruction = f"Explain {topic} in a {style} way."
    response = f"Sure! {topic} is fascinating. Here is a {style} breakdown..." # LLM output goes here
    
    # MLX-LM Format (ChatML-style)
    return {
        "text": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"
    }

# Generate 200 samples
dataset = []
for _ in range(200):
    topic = random.choice(topics)
    style = random.choice(styles)
    dataset.append(generate_sample(topic, style))

# Save as train.jsonl
with open('data/train.jsonl', 'w') as f:
    for entry in dataset:
        f.write(json.dumps(entry) + '\n')

print(f"Generated {len(dataset)} samples in data/train.jsonl")

