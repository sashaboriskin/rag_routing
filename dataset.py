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
        self.data = self.load_or_process_data()

    def load_or_process_data(self):
        if os.path.exists(self.file_path):
            return pd.read_csv(self.file_path)
        
        self.sampling_params = GenerationConfig(**self.cfg.generation_config)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_id).to(self.device)
        
        data = self.preprocess()
        data = self.generate_answers(data)
        data = self.gpt_correctness(data)
        data.to_csv(self.file_path, index=False)
        return data
    
    def preprocess(self):
        self.data = load_dataset(self.data_path)['train'].to_pandas()
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
        try:
            max_new_tokens = df['golden_answer'].apply(len).mean()
        except:
            # if golden_answer is not provided
            max_new_tokens = 100
            
        def generate_answer(question, system_prompt):
            messages = [
                {"role": "user", "content": question},
                {"role": "system", "content": system_prompt}
            ]
            input_data = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            input_data = self.tokenizer(
                input_data, return_tensors="pt", add_special_tokens=False
            ).to(self.device)

            input_len = input_data["input_ids"].shape[1]
            
            generated = self.model.generate(
                input_ids=input_data["input_ids"],
                attention_mask=input_data["attention_mask"],
                max_new_tokens=max_new_tokens,
                generation_config=self.sampling_params,
            )
            answer = self.tokenizer.batch_decode(
                generated[:, input_len:], skip_special_tokens=True
            )[0]
            return answer
        
        for index, row in tqdm(df.iterrows(), total=len(df)):
            df.loc[index, 'our_answer_wo_context'] = generate_answer(
                question=row['question'],
                system_prompt=wo_context_system_prompt()
            )

            df.loc[index, 'our_answer_w_context'] = generate_answer(
                question=w_context_user_prompt(row['question'], row['context']),
                system_prompt=w_context_system_prompt()
            )
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
            file_path='data/squad2.csv',
        )
    
    def load_or_process_data(self):
        if os.path.exists(self.file_path):
            return pd.read_csv(self.file_path)
            
        self.sampling_params = GenerationConfig(**self.cfg.generation_config)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_id).to(self.device)
    
        data = self.preprocess()
        data = self.generate_answers(data)
        data.to_csv(self.file_path, index=False)
        return data
    
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
        self.data = self.data.sample(500, random_state=self.cfg.seed)
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
    
    def preprocess(self):
        self.data = load_dataset(self.data_path)['train'].to_pandas()
        self.data = self.data[['question_text', 'reference', 'contexts']]
        self.data = self.data.rename(columns={
            'question_text': 'question', 
            'reference': 'golden_answer', 
            'contexts': 'context'
        })
        # union if multiple golden answers are given
        self.data['golden_answer'] = self.data['golden_answer'].apply(lambda x: '; '.join(x))
        
        def select_context(contexts):
            for context in contexts:
                if context.get('is_supporting', False):
                    return context['paragraph_text']
            return contexts[-1]['paragraph_text']

        self.data['context'] = self.data['context'].apply(select_context)
        return self.data


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
    def preprocess(self):
        self.data = load_dataset(self.data_path)['train'].to_pandas()
        self.data = self.data[['question_text', 'reference', 'contexts']]
        self.data = self.data.rename(columns={
            'question_text': 'question', 
            'reference': 'golden_answer', 
            'contexts': 'context'
        })
        # union if multiple golden answers are given
        self.data['golden_answer'] = self.data['golden_answer'].apply(lambda x: '; '.join(x))
        
        def select_context(contexts):
            for context in contexts:
                if context.get('is_supporting', False):
                    return context['paragraph_text']
            return contexts[-1]['paragraph_text']

        self.data['context'] = self.data['context'].apply(select_context)
        return self.data


class HotPotQADataset(AbstractDataset):
    def __init__(self):
        super().__init__(
            data_path='VityaVitalich/adaptive_rag_hotpotqa',
            file_path='data/adaptive_rag_hotpotqa.csv'
        )
    def preprocess(self):
        self.data = load_dataset(self.data_path)['train'].to_pandas()
        self.data = self.data[['question_text', 'reference', 'contexts']]
        self.data = self.data.rename(columns={
            'question_text': 'question', 
            'reference': 'golden_answer', 
            'contexts': 'context'
        })
        # union if multiple golden answers are given
        self.data['golden_answer'] = self.data['golden_answer'].apply(lambda x: '; '.join(x))
        
        def select_context(contexts):
            for context in contexts:
                if context.get('is_supporting', False):
                    return context['paragraph_text']
            return contexts[-1]['paragraph_text']

        self.data['context'] = self.data['context'].apply(select_context)
        return self.data

class MusiqueDataset(AbstractDataset):
    def __init__(self):
        super().__init__(
            data_path='VityaVitalich/adaptive_rag_musique',
            file_path='data/adaptive_rag_musique.csv'
        )
    def preprocess(self):
        self.data = load_dataset(self.data_path)['train'].to_pandas()
        self.data = self.data[['question_text', 'reference', 'contexts']]
        self.data = self.data.rename(columns={
            'question_text': 'question', 
            'reference': 'golden_answer', 
            'contexts': 'context'
        })
        # union if multiple golden answers are given
        self.data['golden_answer'] = self.data['golden_answer'].apply(lambda x: '; '.join(x))
        
        def select_context(contexts):
            for context in contexts:
                if context.get('is_supporting', False):
                    return context['paragraph_text']
            return contexts[-1]['paragraph_text']

        self.data['context'] = self.data['context'].apply(select_context)
        return self.data
