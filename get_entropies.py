import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

from lm_polygraph.stat_calculators import EntropyCalculator
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

def layer_wise_entropy(activation_dict, model) -> torch.tensor:
    """
    Given stored activations from each layer (residual stream),
    apply ln_final and unembed to get entropy per layer for 1 token.
    """
    logit_lens = []
    estimator = EntropyCalculator()
    
    for state in range(model.cfg.n_layers):
        resid = activation_dict[f"blocks.{state}.hook_resid_post"] # shape: [batch, seq_len, d_model]
        logits = model.ln_final(resid) # shape: [batch, seq_len, d_model]
        logits = model.unembed(logits) # shape: [batch, seq_len, vocab_size]
        last_token_resid = logits[:, -1, :] # shape: [batch, vocab_size]
        log_probs = F.log_softmax(last_token_resid, dim=-1).detach().cpu().unsqueeze(dim=0) # shape: [1, batch, vocab_size]
        uncertainty = estimator({"greedy_log_probs": np.array(log_probs)})['entropy'][0][0] # shape: n
        logit_lens.append(uncertainty)
        
    return torch.tensor(logit_lens) # shape: [n_layers]
    
def layer_token_wise_entropy(tokens, model, max_new_tokens):
    """
    
    """
    entropies = []
    for step in range(max_new_tokens):
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
        
        # Store the entropies for this step
        entropies.append(layer_wise_entropy(activation_store, model))
        
        # Append the predicted token to the input for the next iteration
        tokens = torch.cat([tokens, next_token_id.to(device)], dim=-1)
        
        if model.to_str_tokens(next_token_id)[0] ==  "<|eot_id|>":
            break
            
    return torch.stack(entropies, dim=0) # shape: [num_new_tokens, n_layers]

def calculate_entropies_in_dataset(dataset, with_context, model, tokenizer, max_new_tokens):
    """

    """
    all_entropies = []
    all_eos = []
    max_tokens = 0
    
    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
        if with_context:
            text = tokenizer.apply_chat_template([
                {"role": "user", "content": w_context_user_prompt(row['question'], row['context'])}, 
                {"role": "system", "content": w_context_system_prompt()}
            ], tokenize=False, add_generation_prompt=True)
        else:
            text = tokenizer.apply_chat_template([
                {"role": "user", "content": row['question']}, 
                {"role": "system", "content": wo_context_system_prompt()}
            ], tokenize=False, add_generation_prompt=True)
            
        tokens = model.to_tokens(text, prepend_bos=False).to(device)
        entropy = layer_token_wise_entropy(tokens, model, max_new_tokens)
        eos_entropy = entropy[-1]  # shape: [n_layers]
        entropy = entropy[:-1] # shape: [num_new_tokens, n_layers]
        
        all_entropies.append(entropy)
        all_eos.append(eos_entropy)
        max_tokens = max(max_tokens, entropy.shape[0])
        
    return all_entropies, all_eos, max_tokens

def agregate_entropy_dataset(dataset, dataset_name, subsample_name, with_context, model, tokenizer, max_new_tokens=100):
    """
    Aggregates entropy data across a dataset with confidence intervals (mean ± std).
    """
    all_entropies, all_eos, max_tokens = calculate_entropies_in_dataset(dataset, with_context, model, tokenizer, max_new_tokens)

    with open(f'data/origin_entropies_wo_agg/all_entropies_{dataset_name}_{subsample_name}.pickle', 'wb') as file:
        pickle.dump(all_entropies, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(f'data/origin_entropies_wo_agg/all_eos_entropies_{dataset_name}_{subsample_name}.pickle', 'wb') as file:
        pickle.dump(all_eos, file, protocol=pickle.HIGHEST_PROTOCOL)

    # Create tensors with NaN for alignment
    padded_entropies = torch.full((len(all_entropies), max_tokens, model.cfg.n_layers), float('nan'))
    
    for i, entropy in enumerate(all_entropies):
        length = entropy.shape[0]
        padded_entropies[i, :length, :] = entropy

    # Calculate mean and standard deviation, ignoring NaNs
    mean_entropies = torch.nanmean(padded_entropies, dim=0)
    std_entropies = torch.from_numpy(np.nanstd(padded_entropies, axis=0))

    # Calculate mean and std for EOS tokens
    aggregated_eos_mean = torch.stack(all_eos).mean(dim=0)
    aggregated_eos_std = torch.stack(all_eos).std(dim=0)
    
    # Convert to a DataFrame with formatted mean ± std
    def format_mean_std(mean_tensor, std_tensor):
        return np.vectorize(lambda mean, std: f"{mean:.4f}±{std:.4f}")(mean_tensor.numpy(), std_tensor.numpy())

    # Prepare formatted DataFrame
    formatted_entropies = format_mean_std(mean_entropies.T, std_entropies.T)
    formatted_eos = format_mean_std(aggregated_eos_mean, aggregated_eos_std)

    # Compute mean and std across layers for each token
    mean_across_layers = torch.nanmean(mean_entropies, dim=0)
    std_across_layers = torch.from_numpy(np.nanstd(mean_entropies, axis=0))
    formatted_mean_across_layers = format_mean_std(mean_across_layers, std_across_layers)

    # Compute mean and std across tokens for each layer
    mean_across_tokens = torch.nanmean(mean_entropies, dim=1)
    std_across_tokens = torch.from_numpy(np.nanstd(mean_entropies, axis=1))
    formatted_mean_across_tokens = format_mean_std(mean_across_tokens, std_across_tokens)
    
    df = pd.DataFrame(formatted_entropies, index=[f"Layer {i+1}" for i in range(model.cfg.n_layers)],
                      columns=[f"Token {i+1}" for i in range(max_tokens)])
    df['EOS'] = formatted_eos
    df['Mean'] = formatted_mean_across_layers
    
    # Add 'Mean' row (average across tokens)
    mean_row = list(formatted_mean_across_tokens) + [format_mean_std(aggregated_eos_mean.mean(), aggregated_eos_std.mean())]
    df.loc['Mean'] = mean_row

    # Compute mean and std across layers for each token (Mean Column)
    mean_across_layers = torch.nanmean(mean_entropies, dim=0)
    std_across_layers = torch.from_numpy(np.nanstd(mean_entropies, axis=0))
    formatted_mean_across_layers = list(format_mean_std(mean_across_layers, std_across_layers))

    layer_token_mean = format_mean_std(mean_across_layers.mean(), std_across_layers.mean())

    formatted_mean_across_layers.append(layer_token_mean)
    df['Mean'] = formatted_mean_across_layers
    
    return df


if __name__ == "__main__":
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
        w_false = dataset[dataset['is_correct_w_context']==0]
        w_true = dataset[dataset['is_correct_w_context']==1]
        wo_false = dataset[dataset['is_correct_wo_context']==0]
        wo_true = dataset[dataset['is_correct_wo_context']==1]

        for subsample, subsample_name in zip(
            (w_false, w_true, wo_false, wo_true), 
            ('w_false', 'w_true', 'wo_false', 'wo_true')
        ):
            if 'wo' in subsample_name:
                test = agregate_entropy_dataset(subsample, dataset_name, subsample_name, False, model, tokenizer, max_new_tokens)
            else:
                test = agregate_entropy_dataset(subsample, dataset_name, subsample_name, True, model, tokenizer, max_new_tokens)
            
            test.to_csv(f'data/tl_mean_std_entropies_datasets/{dataset_name}_{subsample_name}_tl_entropy.csv')
