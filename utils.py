import matplotlib.pyplot as plt
import numpy as np

def transfer_context_prompt(question, context):
    return f'Answer the question {question} based on the given context {context}'
    # In case there is no relevent information in the context, answer than you do not know'

def barplot_uncertainty(method_name, path_to_save, tokens, ue_scores):
    plt.figure(figsize=(12, 6))
    x = np.arange(len(tokens))
    plt.bar(x, ue_scores, tick_label=tokens)
    plt.xticks(rotation=90)
    plt.xlabel("Tokens")
    plt.ylabel("Uncertainty")
    plt.title(f"{method_name} of Each Token")
    plt.tight_layout()
    plt.savefig(path_to_save)
