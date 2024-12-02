from abc import ABC, abstractmethod
import csv
import re
import os
import json

from datasets import load_dataset
import logging
import torch
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from omegaconf import OmegaConf
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


from prompts import (
    w_context_user_prompt,
    w_context_system_prompt,
    wo_context_system_prompt,
    correctness_user_prompt, 
    correctness_system_prompt
)

load_dotenv()

class AbstractDataset(ABC):
    def __init__(self, data_path, file_path):
        self.cfg = OmegaConf.load("config.yaml")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.data_path = data_path
        self.file_path = file_path
        self.data = self._load_or_process_data()

    def _load_or_process_data(self):
        if os.path.exists(self.file_path):
            return pd.read_csv(self.file_path)
        
        self.sampling_params = GenerationConfig(**self.cfg.generation_config)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_id).to(self.device)
        
        data = self.preprocess()
        data = self.generate_answers(data)
        data = self.gpt_correctness(data)
        #data.to_csv(self.file_path, index=False)
        return data
    
    def preprocess(self):
        self.data = load_dataset(self.data_path)['train'].to_pandas().sample(frac=0.05)
        self.data = self.data[['question_text', 'reference', 'gold_context']]
        self.data = self.data.rename(columns={
            'question_text': 'question', 
            'reference': 'golden_answer', 
            'gold_context': 'context'
        })
        # union if multiple golden answers are given
        self.data['golden_answer'] = self.data['golden_answer'].apply(lambda x: '; '.join(x))
        # remove Title: ..., Content: 
        self.data['context'] = self.data['context'].apply(lambda x: self.extract_content(x[0]))
        return self.data
        

    def gpt_correctness(self, df):
        for index, row in tqdm(df.iterrows()):

            is_correct_wo_context = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": correctness_user_prompt(
                        question=row['question'], 
                        golden_answer=row['golden_answer'], 
                        model_answer=row['our_answer_wo_context']
                    )},
                    {"role": "system", "content": correctness_system_prompt()}
                ]
            ).choices[0].message.content

            is_correct_w_context = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": correctness_user_prompt(
                        question=row['question'], 
                        golden_answer=row['golden_answer'], 
                        model_answer=row['our_answer_w_context']
                    )},
                    {"role": "system", "content": correctness_system_prompt()}
                ]
            ).choices[0].message.content

            df.loc[index, 'is_correct_wo_context'] = is_correct_wo_context
            df.loc[index, 'is_correct_w_context'] = is_correct_w_context

        return df
    
    def extract_content(self, text):
        match = re.search(r'Content:\s*(.*)', text)
        return match.group(1) if match else text
    
    def generate_answers(self, df):
        max_new_tokens = df['golden_answer'].apply(len).mean()

        for index, row in tqdm(df.iterrows()):
            our_answer_wo_context = [
                {"role": "user", "content": row['question']}, 
                {"role": "system", "content": wo_context_system_prompt()}
            ]

            our_answer_wo_context = self.tokenizer.apply_chat_template(our_answer_wo_context, add_generation_prompt=True, tokenize=False)
            our_answer_wo_context = self.tokenizer(our_answer_wo_context, return_tensors="pt", add_special_tokens=False).to(self.device)
            our_answer_wo_context = self.tokenizer.batch_decode(self.model.generate(
                **our_answer_wo_context, 
                max_new_tokens=max_new_tokens,
                generation_config=self.sampling_params,
            ))[0]
            match = re.search(r'<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>', our_answer_wo_context, re.DOTALL)
            our_answer_wo_context = match.group(1) if match else our_answer_wo_context
            
            our_asnwer_w_context = [
                {"role": "user", "content": w_context_user_prompt(row['question'], row['context'])}, 
                {"role": "system", "content": w_context_system_prompt()}
            ]
            our_asnwer_w_context = self.tokenizer.apply_chat_template(our_asnwer_w_context, add_generation_prompt=True, tokenize=False)
            our_asnwer_w_context = self.tokenizer(our_asnwer_w_context, return_tensors="pt", add_special_tokens=False).to(self.device)
            our_asnwer_w_context = self.tokenizer.batch_decode(self.model.generate(
                **our_asnwer_w_context, 
                max_new_tokens=max_new_tokens,
                generation_config=self.sampling_params,
            ))[0]
            match = re.search(r'<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>', our_asnwer_w_context, re.DOTALL)
            our_asnwer_w_context = match.group(1) if match else our_asnwer_w_context

            df.loc[index, 'our_answer_wo_context'] = our_answer_wo_context
            df.loc[index, 'our_answer_w_context'] = our_asnwer_w_context

        return df

class OurNQDataset(AbstractDataset):
    def __init__(self):
        super().__init__(
            data_path='cjlovering/natural-questions-short',
            file_path='data/natural-questions-short.csv'
        )

    def preprocess(self):
        self.data = load_dataset(self.data_path)['train'].shuffle(seed=self.cfg.seed).select(range(1000)).to_pandas()
        self.data = self.data[['contexts', 'questions', 'answers']]
        self.data['questions'] = self.data['questions'].apply(lambda x: x[0]['input_text'])
        self.data['answers'] = self.data['answers'].apply(lambda x: x[0]['span_text'])
        self.data = self.data.rename(columns={'questions': 'question', 'answers': 'golden_answer', 'contexts': 'context'})
        return self.data


class Squad2Dataset(AbstractDataset):
    def __init__(self):
        super().__init__(
            data_path='data/squad2/impossible_questions_with_unique_contexts.csv',
            file_path='data/squad2/impossible_questions_with_unique_contexts.csv',
            data=None
        )
        if os.path.exists(self.data_path):
            self.data = pd.read_csv(self.file_path)
        else:
            self.data = self.preprocess()
            self.data = self.generate_answers(self.data)
    
    def preprocess(self):
        if not os.path.exists('data/squad2/train-v2.0.json'):
            raise FileNotFoundError(f"Squad2 json is not found")
        
        with open('data/squad2/train-v2.0.json', 'r') as file:
            data = json.load(file)

        impossible_questions_with_unique_contexts = []
        unique_contexts = set()

        for item in data['data']:
            for paragraph in item['paragraphs']:
                context = paragraph['context']
                if context not in unique_contexts: 
                    for qas in paragraph['qas']:
                        if qas.get('is_impossible', False):
                            impossible_questions_with_unique_contexts.append({
                                "question": qas['question'],
                                "context": context
                            })
                            unique_contexts.add(context)
                            break

        with open(self.data_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['question', 'context'])
            writer.writeheader()
            for item in impossible_questions_with_unique_contexts:
                writer.writerow(item)
        
        self.data = pd.read_csv(self.data_path)
        return self.data


class NQDataset(AbstractDataset):
    def __init__(self):
        super().__init__(
            data_path='VityaVitalich/adaptive_rag_natural_questions',
            file_path='data/adaptive_rag_natural_questions.csv'
        )


class TriviaQADataset(AbstractDataset):
    def __init__(self):
        super().__init__(
            data_path='VityaVitalich/adaptive_rag_trivia_qa',
            file_path='data/adaptive_rag_trivia_qa.csv'
        )


class SquadDataset(AbstractDataset):
    def __init__(self):
        super().__init__(
            data_path='VityaVitalich/adaptive_rag_squad',
            file_path='data/adaptive_rag_squad.csv'
        )


class WikiMultiHopQADataset(AbstractDataset):
    def __init__(self):
        super().__init__(
            data_path='VityaVitalich/adaptive_rag_2wikimultihopqa',
            file_path='data/adaptive_rag_2wikimultihopqa.csv'
        )


class HotPotQADataset(AbstractDataset):
    def __init__(self):
        super().__init__(
            data_path='VityaVitalich/adaptive_rag_hotpotqa',
            file_path='data/adaptive_rag_hotpotqa.csv'
        )


class MusiqueDataset(AbstractDataset):
    def __init__(self):
        super().__init__(
            data_path='VityaVitalich/adaptive_rag_musique',
            file_path='data/adaptive_rag_musique.csv'
        )


if __name__ == '__main__':
    df = NQDataset().data
    print(df)