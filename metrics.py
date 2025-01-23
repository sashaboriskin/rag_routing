import string
import re
from collections import Counter
import re
import numpy as np

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))
    
def InAcc_score(prediction, ground_truth):
    if ground_truth in prediction:
        return 1
    return 0

def f1(decoded_preds, decoded_labels):
    f1_all = []
    for prediction, answers in zip(decoded_preds, decoded_labels):
        if type(answers) == list:
            if len(answers) == 0:
                return 0
            f1_all.append(np.max([qa_f1_score(prediction, gt)
                          for gt in answers]))
        else:
            f1_all.append(qa_f1_score(prediction, answers))
    return np.mean(f1_all)

def qa_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match(prediction, ground_truths):
    matches = [exact_match_score(pred, label) for pred, label in zip(prediction, ground_truths)]
    return matches.count(1)/prediction.shape[0]

def InAcc(prediction, ground_truths):
    matches = [InAcc_score(pred, label) for pred, label in zip(prediction, ground_truths)]
    return matches.count(1)/prediction.shape[0]
