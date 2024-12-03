import json

file_path = 'data/wikimultihopqa/dev_500_subsampled.jsonl'

all_have_supporting = True

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        has_supporting = any(context.get('is_supporting', False) for context in data.get('contexts', []))
        if not has_supporting:
            print(f"Строка с question_id={data['question_id']} не имеет is_supporting=True")
            all_have_supporting = False

if all_have_supporting:
    print("Во всех строках есть хотя бы один paragraph_text с is_supporting=True.")
else:
    print("Не во всех строках есть paragraph_text с is_supporting=True.")