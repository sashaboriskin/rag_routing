from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.estimators import (
    MeanTokenEntropy, 
    TokenEntropy,
    MaximumSequenceProbability,
    ClaimConditionedProbability, 
    TokenSAR,
)
from lm_polygraph.utils.manager import estimate_uncertainty
import pandas as pd
from tqdm import tqdm

from utils import transfer_context_prompt, barplot_uncertainty


model_path = "/share/nlp/chitchat/models/Llama-3.2-3B-Instruct/"
sampling_params = {
    'temperature':0.6,
    'top_k':50, 
    'top_p':0.9,
    'do_sample':False,
    'num_beams':1,
    'presence_penalty':0.0,
    'repetition_penalty':1.0, 
    'generate_until':(),
    'allow_newlines':True,
}

model = WhiteboxModel.from_pretrained(
    model_path=model_path, 
    generation_params=sampling_params,
    device_map="auto"
)

TokenEntropy_method = TokenEntropy()
MeanTokenEntropy_method = MeanTokenEntropy()
MaximumSequenceProbability_method = MaximumSequenceProbability()
ClaimConditionedProbability_method = ClaimConditionedProbability()
TokenSAR_method = TokenSAR()

name2method = {
    'TokenEntropy': TokenEntropy_method,
    'MeanTokenEntropy': MeanTokenEntropy_method,
    #'MaximumSequenceProbability': MaximumSequenceProbability_method,
    #'ClaimConditionedProbability': ClaimConditionedProbability_method,
    #'TokenSAR': TokenSAR_method,
}

eval_dataset = pd.read_csv('data/rag_routing_eval_dataset_context_perturbations.csv')

for index, row in tqdm(eval_dataset.iterrows()):
    question = row['question']
    context = row['context_perturbations']
    
    for method_name in name2method.keys():
        print(f"Processing question {index}, method {method_name}")
        
        result_without_context = estimate_uncertainty(
            model, 
            name2method[method_name], 
            input_text=question,
            clean_tokens_in_output=True
        )
        our_answer_without_context, our_tokens_without_context, ue_score_without_context = result_without_context.generation_text, result_without_context.generation_tokens, result_without_context.uncertainty

        result_with_context = estimate_uncertainty(
            model, 
            name2method[method_name], 
            input_text=transfer_context_prompt(question, context),
            clean_tokens_in_output=True
        )
        our_answer_with_context, our_tokens_with_context, ue_score_with_context = result_with_context.generation_text, result_with_context.generation_tokens, result_with_context.uncertainty

        # scalar ue scores
        if method_name not in ('TokenEntropy', 'MaximumTokenProbability'):
            eval_dataset.loc[index, f'{method_name}_without_context'] = round(ue_score_without_context, 4)
            eval_dataset.loc[index, f'{method_name}_with_context'] = round(ue_score_with_context, 4)
            eval_dataset.loc[index, f'our_answer_without_context'] = our_answer_without_context
            eval_dataset.loc[index, f'our_answer_with_context'] = our_answer_with_context
            print(f"{method_name}: {ue_score_without_context} (without context), {ue_score_with_context} (with context)")

        # token based ue scores
        else:
            all_decoded_tokens_without_context = [model.tokenizer.decode(our_tokens_without_context[ind]) for ind in range(len(our_tokens_without_context))]
            all_decoded_tokens_with_context = [model.tokenizer.decode(our_tokens_with_context[ind]) for ind in range(len(our_tokens_with_context))]
            
            barplot_uncertainty(
                method_name, 
                f'data/ue_graphs_perturbations/{index}_{method_name}_without_context.jpg', 
                all_decoded_tokens_without_context, 
                ue_score_without_context
            )
            
            barplot_uncertainty(
                method_name, 
                f'data/ue_graphs_perturbations/{index}_{method_name}_with_context.jpg', 
                all_decoded_tokens_with_context, 
                ue_score_with_context
            )

eval_dataset.to_csv('data/ue_scores_eval_dataset_perturbations.csv', index=False)
