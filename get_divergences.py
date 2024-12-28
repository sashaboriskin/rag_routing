import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

from prompts import w_context_user_prompt, w_context_system_prompt, wo_context_system_prompt
from dataset import (
    NQDataset,
    WikiMultiHopQADataset, 
    HotPotQADataset, 
    MusiqueDataset
)

def store_activations(activation_dict):
    def hook_fn(activation, hook):
        activation_dict[hook.name] = activation.detach()
    return hook_fn

def get_logit_lens(activation_dict, model) -> torch.tensor:
    """
    Given stored activations from each layer (residual stream),
    apply ln_final and unembed to get logits per layer.
    """
    logit_lens = []

    for state in range(model.cfg.n_layers):
        resid = activation_dict[f"blocks.{state}.hook_resid_post"] # shape: [batch, seq_len, d_model]
        logits = model.ln_final(resid) # shape: [batch, seq_len, d_model]
        logits = model.unembed(logits) # shape: [batch, seq_len, vocab_size]
        last_token_resid = logits[:, -1, :] # shape: [batch, vocab_size]
        logit_lens.append(last_token_resid.detach().cpu())

    return torch.stack(logit_lens, dim=0) # shape: [n_layers, batch, vocab_size]
    
def get_logits(tokens, model, num_new_tokens):
    all_logit_lens = []

    for step in range(num_new_tokens):
        activation_store = {}
        
        with torch.no_grad():
            # Run model with hooks to capture residual activations
            logits = model.run_with_hooks(
                tokens,
                return_type="logits",
                fwd_hooks=[(f"blocks.{i}.hook_resid_post", store_activations(activation_store)) 
                           for i in range(model.cfg.n_layers)],
            )
            
        logits = logits.detach().cpu()
    
        # Get the final next-token prediction from the model
        final_logits = logits[:, -1, :]
        next_token_id = torch.argmax(final_logits, dim=-1, keepdim=True)
        
        # Store the logit lens for this step
        all_logit_lens.append(get_logit_lens(activation_store, model))
        
        # Append the predicted token to the input for the next iteration
        tokens = torch.cat([tokens, next_token_id.to(device)], dim=-1)

        if model.to_str_tokens(next_token_id)[0] ==  "<|eot_id|>":
            break
        
    return torch.stack(all_logit_lens, dim=0) # shape: [num_new_tokens, n_layers, batch, vocab_size]

def calculate_js_divergence(softmax1, softmax2):
    """
    Calculate the Jensen-Shannon divergence between two probability distributions.
    """
    M = 0.5 * (softmax1 + softmax2)  # Average distribution
    kl1 = F.kl_div(softmax1.log(), M, reduction='batchmean')  # KL divergence for softmax1
    kl2 = F.kl_div(softmax2.log(), M, reduction='batchmean')  # KL divergence for softmax2
    js_div = 0.5 * (kl1 + kl2)  # JS divergence
    return js_div

def calculate_divs_in_dataset(dataset, model, tokenizer, max_new_tokens=100) -> (list[torch.tensor], list[torch.tensor]):
    all_div_matrices = [] # To store layer-wise and token-wise divergence matrixs
    all_eot_div = []  # To store divergence values for <|eot_id|>

    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
        text_w_context = tokenizer.apply_chat_template([
            {"role": "user", "content": w_context_user_prompt(row['question'], row['context'])},
            {"role": "system", "content": w_context_system_prompt()},
        ], tokenize=False, add_generation_prompt=True)
   
        text_w_context = model.to_tokens(text_w_context, prepend_bos=False).to(device)
        logits = get_logits(text_w_context, model, max_new_tokens) # shape: [num_new_tokens, n_layers, batch, vocab_size]
        logits_w_context = logits[:-1]
        eot_logits_w_context = logits[-1] # shape: [1, n_layers, batch, vocab_size]
        
        text_wo_context = tokenizer.apply_chat_template([
            {"role": "user", "content": row['question']},
            {"role": "system", "content": wo_context_system_prompt()},
        ], tokenize=False, add_generation_prompt=True)
        
        text_wo_context = model.to_tokens(text_wo_context, prepend_bos=False).to(device)
        logits = get_logits(text_wo_context, model, max_new_tokens)[:-1] # shape: [num_new_tokens, n_layers, batch, vocab_size]
        logits_wo_context = logits[:-1]
        eot_logits_wo_context = logits[-1] # shape: [1, n_layers, batch, vocab_size]
            
        min_length = min(logits_w_context.shape[0], logits_wo_context.shape[0])
        logits_w_context = logits_w_context[:min_length] # shape: [min_num_new_tokens, n_layers, batch, vocab_size]
        logits_wo_context = logits_wo_context[:min_length] # shape: [min_num_new_tokens, n_layers, batch, vocab_size]

        # Calculate JS/KL layer-wise and token-wise divergence 
        div_matrix = np.zeros((model.cfg.n_layers, min_length)) #shape: [num_layers, min_num_new_tokens]
        for token_idx in range(min_length):
            for layer_idx in range(model.cfg.n_layers):
                logits_w = logits_w_context[token_idx][layer_idx] # shape: [batch, vocab_size]
                logits_wo = logits_wo_context[token_idx][layer_idx] # shape: [batch, vocab_size]
                
                # Softmax probabilities
                softmax_w = F.softmax(logits_w, dim=-1) # shape: [batch, vocab_size]
                softmax_wo = F.softmax(logits_wo, dim=-1) # shape: [batch, vocab_size]

                kl_div = calculate_js_divergence(softmax_w, softmax_wo)
                #kl_div = F.kl_div(softmax_w, softmax_wo, reduction='batchmean')
                div_matrix[layer_idx, token_idx] = kl_div.item()
            
        all_div_matrices.append(torch.tensor(div_matrix))

        # Calculate JS/KL divergence for <|eot_id|>
        all_eot_div.append(torch.tensor([
            calculate_js_divergence(
                F.softmax(eot_logits_w_context[layer_idx], dim=-1),
                F.softmax(eot_logits_wo_context[layer_idx], dim=-1),
            ).item()
            for layer_idx in range(model.cfg.n_layers)
        ])) # shape: [num_layers]
        
    return all_div_matrices, all_eot_div

def agregate_div_dataset(dataset, model, tokenizer, max_new_tokens=100):
    all_div_matrices, all_eot_div = calculate_divs_in_dataset(dataset, model, tokenizer, max_new_tokens)
    
    max_tokens = max(matrix.shape[1] for matrix in all_div_matrices)
    aggregated_matrix = np.zeros((model.cfg.n_layers, max_tokens))
    count_matrix = np.zeros((model.cfg.n_layers, max_tokens))
    
    for matrix in all_div_matrices:
        for layer_idx in range(model.cfg.n_layers):
            for token_idx in range(matrix.shape[1]):
                aggregated_matrix[layer_idx, token_idx] += matrix[layer_idx, token_idx]
                count_matrix[layer_idx, token_idx] += 1
                
    aggregated_matrix /= np.maximum(count_matrix, 1)
    
    # Add <|eot_id|> divergences as the last column
    eot_kl_column = np.mean(all_eot_div, axis=0).reshape(-1, 1)
    aggregated_matrix = np.hstack([aggregated_matrix, eot_kl_column])
    
    # Calculate mean for each column
    token_means = np.mean(aggregated_matrix, axis=0)
    
    # Create a DataFrame
    columns = [f"{i+1}_token" for i in range(max_tokens)] + ["<|eot_id|>"]
    index = [f"Layer_{i+1}" for i in range(model.cfg.n_layers)] + ["Mean"]
    aggregated_matrix = np.vstack([aggregated_matrix, token_means])

    df = pd.DataFrame(aggregated_matrix, columns=columns, index=index)
    df.index.name = "Layer"
    return df

if __name__ == '__main__':
    nq_dataset = NQDataset().data
    wiki_multi_dataset = WikiMultiHopQADataset().data
    hot_pot_dataset = HotPotQADataset().data
    musique_dataset = MusiqueDataset().data

    cfg = OmegaConf.load("config.yaml")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    model = HookedTransformer.from_pretrained(
        cfg.model_id,
        device=device,
        tokenizer=tokenizer,
    )
    model.set_use_attn_result(True)
    model.eval()

    for dataset, dataset_name in zip(
        (nq_dataset, wiki_multi_dataset, hot_pot_dataset, musique_dataset),
        ('nq_dataset', 'wiki_multi_dataset', 'hot_pot_dataset', 'musique_dataset')
    ):
        print(f'Preproccesing {dataset_name}')
        
        max_new_tokens = int(dataset['reference'].apply(len).mean())
        wo0_w1 = dataset[dataset['is_correct_w_context']==1][dataset['is_correct_wo_context']==0]
        wo0_w0 = dataset[dataset['is_correct_w_context']==0][dataset['is_correct_wo_context']==0]
        wo1_w0 = dataset[dataset['is_correct_w_context']==0][dataset['is_correct_wo_context']==1]
        wo1_w1 = dataset[dataset['is_correct_w_context']==1][dataset['is_correct_wo_context']==1]
        for subsample, subsample_name in zip(
            (wo0_w1, wo0_w0, wo1_w0, wo1_w1), 
            ('wo0_w1', 'wo0_w0', 'wo1_w0', 'wo1_w1')
        ):
            test = agregate_div_dataset(subsample, model, tokenizer, max_new_tokens=max_new_tokens)    
            test.to_csv(f'data/tl_divs_datasets/{dataset_name}_{subsample_name}_tl_div.csv')
