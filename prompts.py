def w_context_user_prompt(question, context):
    return f'Context: "{context}" Question: "{question}"'

def w_context_system_prompt():
    return 'You are a helpful assistant who gives only factually correct short answers. Answer the question based on the given context. If there is no relevant information in the context, respond with "I do not know". The question and context are given below.'

def wo_context_system_prompt():
    return 'You are a helpful assistant who gives only factually correct short answers.'

def correctness_system_prompt():
    return """You will be given a user query, a reference answer, and a model's response. Your task is to evaluate the model’s response only in relation to the reference answer. Focus on the accuracy and completeness of the information in the model’s response compared to the reference. Ignore stylistic differences or minor rephrasings

Evaluation criteria:

Score 0 if the model's answer is entirely incorrect or no answer is provided or lacks information or or includes extraneous details not relevant to the query. 
Score 1 if the model’s answer is entirely correct and sufficiently answers the query without unnecessary information.
Provide your answer in the format: Score.

Your evaluation will be used to measure consistency and accuracy."""

def correctness_user_prompt(question, golden_answer, model_answer):
    return"""Please evaluate the model's response based on its accuracy compared to the reference answer, using one of the following scores:

0: Incorrect, no response, lacks information
1: Completely correct
Format: Score.

Question: {question} 
Reference answers: {golden_answer} 
Model Answer: {model_answer}"""
