"""
This script extracts supportive contexts from the Adaptive RAG datasets and save them in a CSV file.
You can find preprocessed datasets here:
https://huggingface.co/collections/aboriskin/adaptive-rag-supporting-context-674f1251dcbba84f93b4e9d1
"""

import json
import os
import csv

def extract_supportive_contexts(dataset_name, base_path):
    data_list = []
    
    for subset in ['dev_500', 'test']:
        file_path = os.path.join(base_path, dataset_name, f'{subset}_subsampled.jsonl')

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                data.pop('reasoning_steps', None)

                # Union if multiple reference answers are given
                reference = ';'.join(
                    span for obj in data.get('answers_objects', []) for span in obj.get('spans', [])
                )
                
                # Extract each context as a separate row
                for context in data.get('contexts', []):
                    data_list.append({
                        'question_id': data.get('question_id'),
                        'question': data.get('question_text'),
                        'reference': reference,
                        'context': context.get('paragraph_text', ''),
                        'split': subset,
                        'is_supportive': context.get('is_supporting', False)
                    })

    fieldnames = ['question_id', 'question', 'reference', 'context', 'split', 'is_supportive']
    with open(f'adaptive_rag_{dataset_name}.csv', 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_list)

def main():
    base_path = 'adaptive_rag_raw/processed_data'
    datasets = ['2wikimultihopqa', 'hotpotqa', 'musique', 'nq']

    for dataset_name in datasets:
        extract_supportive_contexts(dataset_name, base_path)

if __name__ == "__main__":
    main()
