# Create a dedicated directory and virtual environment
```
mkdir ml-mac && cd ml-mac
python3 -m venv .venv
source .venv/bin/activate
```

# Install Apple's MLX library
```
pip install -U mlx-lm huggingface_hub
```
For an M2 (likely 8GB, 16GB, or 24GB RAM), Llama 3.2 1B or 3B are excellent starting points.  
Log in to Hugging Face to access Llama models (get a token from hf.co/settings/tokens)
```
huggingface-cli login
```

# Prepare Your Data
MLX expects data in a specific .jsonl format. Create a folder named data and a file named train.jsonl inside it: 
```
{
  "text": "<|begin_of_text|>
                <|start_header_id|>
                    user
                <|end_header_id|>\n\n
                Write a bio for a software engineer.
           <|eot_id|>
                <|start_header_id|>
                     assistant
                <|end_header_id|>\n\n
                I am a developer who loves building local AI tools on Mac...
           <|eot_id|>"
}
```
Generating a few hundred high-quality examples manually is tedious. In 2026, the standard way to do this is through Synthetic Data Generation.  
You can use a "Teacher" model (like GPT-4o, Claude 3.5, or even a large local model like Llama 3.3 70B if you have the hardware) to generate variations of instructions and responses.  
The "Seed & Expand" Python Script: trainDataGen.py  
This script uses a small "seed" list of topics and asks an LLM to generate diverse instruction-response pairs for each. You can run this using a local LLM (via Ollama) or an API.  
The training data is created in data/train.jsonl  
```
/your-project-folder
 ├── data/
 │   ├── train.jsonl
 │   └── valid.jsonl  <-- (Add this)
 Run SplitData.py to split Training Data to 90:10 Training and Validation
```

MLX makes this a one-line command. We will use LoRA (Low-Rank Adaptation), which only trains a small "adapter" layer, keeping the main model frozen to save memory.
```
python -m mlx_lm.lora \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --data ./data \
  --train \
  --iters 500 \
  --batch-size 4 \
  --num-layers 16 \
  --val-batches 10
# --iters: How many steps to train (start with 500-1000).
# --num-layers: Tells MLX to only train specific parts of the model.
```

# Output
```
Fetching 6 files: 100%|██████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 89877.94it/s]  
Loading datasets  
Training  
Trainable parameters: 0.216% (6.947M/3212.750M)  
Starting training..., iters: 500  
Calculating loss...: 100%|██████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:02<00:00,  1.87it/s]  
Iter 1: Val loss 5.946, Val took 2.684s  
Iter 10: Train loss 3.636, Learning Rate 1.000e-05, It/sec 1.227, Tokens/sec 192.200, Trained Tokens 1566, Peak mem 3.315 GB  
Iter 20: Train loss 0.933, Learning Rate 1.000e-05, It/sec 1.459, Tokens/sec 230.525, Trained Tokens 3146, Peak mem 3.315 GB  
Iter 30: Train loss 0.713, Learning Rate 1.000e-05, It/sec 1.455, Tokens/sec 229.083, Trained Tokens 4720, Peak mem 3.315 GB  
Iter 40: Train loss 0.629, Learning Rate 1.000e-05, It/sec 1.482, Tokens/sec 234.164, Trained Tokens 6300, Peak mem 3.315 GB  
Iter 50: Train loss 0.555, Learning Rate 1.000e-05, It/sec 1.476, Tokens/sec 231.404, Trained Tokens 7868, Peak mem 3.315 GB  
Iter 60: Train loss 0.406, Learning Rate 1.000e-05, It/sec 1.482, Tokens/sec 234.123, Trained Tokens 9448, Peak mem 3.315 GB  
Iter 70: Train loss 0.259, Learning Rate 1.000e-05, It/sec 1.482, Tokens/sec 232.305, Trained Tokens 11016, Peak mem 3.315 GB  
Iter 80: Train loss 0.164, Learning Rate 1.000e-05, It/sec 1.474, Tokens/sec 230.484, Trained Tokens 12580, Peak mem 3.315 GB  
Iter 90: Train loss 0.127, Learning Rate 1.000e-05, It/sec 1.481, Tokens/sec 233.372, Trained Tokens 14156, Peak mem 3.315 GB  
Iter 100: Train loss 0.118, Learning Rate 1.000e-05, It/sec 1.480, Tokens/sec 233.302, Trained Tokens 15732, Peak mem 3.315 GB  
Iter 100: Saved adapter weights to adapters/adapters.safetensors and adapters/0000100_adapters.safetensors.  
Iter 110: Train loss 0.115, Learning Rate 1.000e-05, It/sec 1.468, Tokens/sec 229.662, Trained Tokens 17296, Peak mem 3.339 GB  
Iter 120: Train loss 0.117, Learning Rate 1.000e-05, It/sec 1.479, Tokens/sec 234.528, Trained Tokens 18882, Peak mem 3.339 GB  
Iter 130: Train loss 0.112, Learning Rate 1.000e-05, It/sec 1.480, Tokens/sec 232.429, Trained Tokens 20452, Peak mem 3.339 GB  
Iter 140: Train loss 0.106, Learning Rate 1.000e-05, It/sec 1.471, Tokens/sec 230.385, Trained Tokens 22018, Peak mem 3.339 GB  
Iter 150: Train loss 0.109, Learning Rate 1.000e-05, It/sec 1.480, Tokens/sec 235.073, Trained Tokens 23606, Peak mem 3.339 GB  
Iter 160: Train loss 0.110, Learning Rate 1.000e-05, It/sec 1.455, Tokens/sec 226.966, Trained Tokens 25166, Peak mem 3.339 GB  
Iter 170: Train loss 0.109, Learning Rate 1.000e-05, It/sec 1.460, Tokens/sec 228.566, Trained Tokens 26732, Peak mem 3.339 GB  
Iter 180: Train loss 0.111, Learning Rate 1.000e-05, It/sec 1.479, Tokens/sec 233.668, Trained Tokens 28312, Peak mem 3.339 GB  
Iter 190: Train loss 0.112, Learning Rate 1.000e-05, It/sec 1.478, Tokens/sec 233.577, Trained Tokens 29892, Peak mem 3.339 GB  
Calculating loss...: 100%|██████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:02<00:00,  2.49it/s]  
Iter 200: Val loss 0.113, Val took 2.012s  
Iter 200: Train loss 0.109, Learning Rate 1.000e-05, It/sec 1.478, Tokens/sec 232.678, Trained Tokens 31466, Peak mem 3.339 GB  
Iter 200: Saved adapter weights to adapters/adapters.safetensors and adapters/0000200_adapters.safetensors.  
Iter 210: Train loss 0.111, Learning Rate 1.000e-05, It/sec 1.466, Tokens/sec 229.803, Trained Tokens 33034, Peak mem 3.339 GB  
Iter 220: Train loss 0.108, Learning Rate 1.000e-05, It/sec 1.479, Tokens/sec 231.587, Trained Tokens 34600, Peak mem 3.339 GB  
Iter 230: Train loss 0.105, Learning Rate 1.000e-05, It/sec 1.477, Tokens/sec 233.907, Trained Tokens 36184, Peak mem 3.339 GB  
Iter 240: Train loss 0.105, Learning Rate 1.000e-05, It/sec 1.468, Tokens/sec 232.892, Trained Tokens 37770, Peak mem 3.339 GB  
Iter 250: Train loss 0.107, Learning Rate 1.000e-05, It/sec 1.472, Tokens/sec 228.998, Trained Tokens 39326, Peak mem 3.339 GB  
```

# Step 5: Test and Fuse
Once finished, you’ll have an safetensors file in adapter folder. You can test it immediately:
```
python -m mlx_lm generate \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --adapter-path adapters/ \
  --prompt "Your custom prompt here"
```

If you're happy with it, you can fuse the adapter into the main model so you can use it in apps like Ollama or LM Studio:
```
python -m mlx_lm.fuse \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --adapter-path adapters/ \
  --save-path ./my-finetuned-llama
```

# Step 6: Start using it
```
python -m mlx_lm generate   --model my-finetuned-llama  --prompt "Write a bio for Software Engineer"
==========
Here's a sample bio for a Software Engineer:  

"Hi, I'm [Your Name], a highly motivated and detail-oriented Software Engineer with a passion for designing and developing innovative software solutions. With [Number] years of experience in the field, I have developed a strong foundation in programming languages, data structures, and software development methodologies.  

Throughout my career, I have worked on a wide range of projects, from mobile apps to web applications, and have a proven track record of delivering high-quality software products  
==========
Prompt: 41 tokens, 193.201 tokens-per-sec  
Generation: 100 tokens, 90.310 tokens-per-sec  
Peak memory: 1.944 GB  
```
By default max-tokens is 100.
```
--max-tokens 500
```
