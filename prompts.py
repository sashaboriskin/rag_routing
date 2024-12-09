def w_context_user_prompt(question, context):
    return f'Context: "{context}" Question: "{question}"'

def w_context_system_prompt():
    return 'You are a helpful assistant. Provide extremely short and factually correct answers based only on the given context. If the context does not contain relevant information, respond with "I do not know". Limit your answer to just a few words. Avoid explanations or additional details.'

def wo_context_system_prompt():
    return 'You are a helpful assistant. Your task is to provide extremely short and factual answers. Respond with just the key fact, name, date, or relevant information in no more than a few words. Avoid any explanations, context, or additional details.'

def correctness_system_prompt():
    return """You will be given a user query, a reference answer, and a model's response. Your task is to evaluate the model’s response only in relation to the reference answer. Focus solely on the accuracy and completeness of the information in the model’s response compared to the reference. Ignore stylistic differences, minor rephrasings, or formatting.

Evaluation criteria:

- Score 0: If the model's answer is entirely incorrect, incomplete, lacks information, or includes extraneous details not relevant to the query.
- Score 1: If the model's answer is entirely correct and sufficiently answers the query without unnecessary information.

Provide your answer strictly as a single digit: either 0 or 1. Do not include any additional text, explanation, or formatting."""


def correctness_user_prompt(question, golden_answer, model_answer):
    return f"""Evaluate the model's response using the following criteria:

0: Incorrect, incomplete, lacks information, or includes irrelevant details.
1: Correct, complete, and sufficiently answers the query without unnecessary details.

Respond strictly with a single digit: 0 or 1.

Question: {question}
Reference answers: {golden_answer}
Model Answer: {model_answer}"""