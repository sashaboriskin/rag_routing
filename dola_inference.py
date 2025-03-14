import pandas as pd
import torch
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from omegaconf import OmegaConf
from dataset import (
    NQDataset,
    WikiMultiHopQADataset, 
    HotPotQADataset, 
    MusiqueDataset
)
from prompts import w_context_user_prompt, w_context_system_prompt, wo_context_system_prompt

cfg = OmegaConf.load("config.yaml")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

output_dir = "data/dola"
os.makedirs(output_dir, exist_ok=True)

datasets = {
    "nq": NQDataset().data,
    "wiki_multi": WikiMultiHopQADataset().data,
    "hot_pot": HotPotQADataset().data,
    "musique": MusiqueDataset().data
}

tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token 
    
model = AutoModelForCausalLM.from_pretrained(cfg.model_id).to(device)
generation_config = GenerationConfig(**cfg.generation_config)

def generate_answers(df, name, tokenizer, model, generation_config, device):
    has_context = 'context' in df.columns
    use_context = has_context and not hasattr(generation_config, 'dola_layers')
    
    input_strings = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing inputs"):
        if use_context:
            system_prompt = w_context_system_prompt()
            user_content = w_context_user_prompt(row['context'], row['question'])
        else:
            system_prompt = wo_context_system_prompt()
            user_content = row['question']
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        input_str = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        input_strings.append(input_str)
    
    inputs = tokenizer(
        input_strings,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
        pad_to_multiple_of=8
    ).to(device)

    avg_answer_length = df['reference'].str.len().mean()
    max_new_tokens = int(avg_answer_length * 1.2)  # Add 20% buffer

    batch_size = 8  # Adjust 
    generated_answers = []
    
    for i in tqdm(range(0, len(inputs.input_ids), batch_size), desc="Generating"):
        batch = {
            "input_ids": inputs.input_ids[i:i+batch_size],
            "attention_mask": inputs.attention_mask[i:i+batch_size]
        }
        
        outputs = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            generation_config=generation_config,
        )
        
        input_lens = batch["attention_mask"].sum(dim=1)
        answers = [
            tokenizer.decode(
                output[input_len:], # Something is wrong with the lengths here
                skip_special_tokens=True
            ).split("assistant")[-1] # Ew
            .replace("\n", "").strip() # Ew
                     
            for output, input_len in zip(outputs, input_lens)
        ]
        generated_answers.extend(answers)

    df["our_answer_dola"] = generated_answers

    output_path = os.path.join(output_dir, f"{name}_with_dola.csv")
    df.to_csv(output_path, index=False)
    
    print(f"Saved {name} results to {output_path}")
    
    return df

for name, dataset in datasets.items():
    print(f"Processing {name} dataset...")
    datasets[name] = generate_answers(
        dataset, name, tokenizer, model, generation_config, device
    )