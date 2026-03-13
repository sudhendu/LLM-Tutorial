import json

with open("data/train.jsonl", "r") as f:
    lines = f.readlines()

# Split: 90% training, 10% validation
split_idx = int(len(lines) * 0.9)
train_data = lines[:split_idx]
valid_data = lines[split_idx:]

with open("data/train.jsonl", "w") as f:
    f.writelines(train_data)

with open("data/valid.jsonl", "w") as f:
    f.writelines(valid_data)

print(f"Created train.jsonl ({len(train_data)} lines) and valid.jsonl ({len(valid_data)} lines)")

