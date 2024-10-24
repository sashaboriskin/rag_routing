from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.estimators import TokenSAR, TokenEntropy
from lm_polygraph.utils.manager import estimate_uncertainty
from utils import transfer_context_prompt, barplot_uncertainty
from transformers import AutoTokenizer

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

#TokenSAR_method = TokenSAR()
TokenEntropy_method = TokenEntropy()

question = 'What is considered the start of the modern comics in Japan?'

clean_entropy = estimate_uncertainty(
    model, 
    TokenEntropy_method, 
    input_text=question,
    clean_tokens_in_output=True
)

entropy = estimate_uncertainty(
    model, 
    TokenEntropy_method, 
    input_text=question,
    clean_tokens_in_output=False
)

print(len(clean_entropy.generation_tokens))

print(len(entropy.generation_tokens))


# our_answer_without_context, our_tokens_without_context, ue_score_without_context = result_without_context.generation_text, result_without_context.generation_tokens, result_without_context.uncertainty
# all_decoded_tokens_without_context = [model.tokenizer.decode(our_tokens_without_context[ind]) for ind in range(len(our_tokens_without_context))]
# all_decoded_tokens_without_context = all_decoded_tokens_without_context[1:]

# print(all_decoded_tokens_without_context)
# print('ffffffff')
# print(our_answer_without_context)
# print(len(all_decoded_tokens_without_context))
# print(ue_score_without_context)

# barplot_uncertainty(
#     'test', 
#     'test_without_context.jpg', 
#     all_decoded_tokens_without_context, 
#     ue_score_without_context
# )
# print(len(all_decoded_tokens_without_context))
# print(our_answer_without_context)
# print('FFFFF')
# print(type(all_decoded_tokens_without_context))
# print(all_decoded_tokens_without_context[-1])
# print(all_decoded_tokens_without_context[-2])
