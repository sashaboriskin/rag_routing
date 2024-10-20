import matplotlib.pyplot as plt

def transfer_context_prompt(question, context):
    return f'Answer the question {question} based on the given context {context}. Give only answer without additional information.'

def barplot_uncertainty(method_name, path_to_save, tokens, ue_scores):
    plt.figure(figsize=(12, 6))
    plt.bar(tokens, ue_scores)
    plt.xticks(rotation=90)
    plt.xlabel("Tokens")
    plt.ylabel("Uncertainty")
    plt.title(f"{method_name} of Each Token")
    plt.tight_layout()
    plt.savefig(path_to_save)
