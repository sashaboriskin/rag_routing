def transfer_context_prompt(question, context):
    return f'Answer the question {question} based on the given context {context}. If there is no relevant information in the context, respond with "I do not know".' 

def assistant_system_prompt():
    return 'You are a helpful assistant who gives only factually correct short answers without additional information'
