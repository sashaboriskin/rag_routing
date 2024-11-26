import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

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

def heatmap_uncertainty(df, title, figsize=(16, 8)):
    plt.figure(figsize=figsize)
    ax = sns.heatmap(df, annot=True, cmap="Purples", cbar=True, linewidths=0.5, linecolor='black')
    plt.title(title, fontsize=16, fontweight='bold')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    plt.xticks(rotation=90, ha='center', fontsize=10)
    plt.ylabel("layer", fontsize=14, rotation=0, labelpad=30)
    plt.yticks(rotation=0, fontsize=10)
    plt.show()
