from abc import ABC, abstractmethod
import os

from datasets import load_dataset
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
    def __init__(self, hf_path, file_path):
        self.cfg = OmegaConf.load("config.yaml")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.hf_path = hf_path
        self.file_path = file_path
        self.data = self.load_or_process_data()

    def load_or_process_data(self):
        if os.path.exists(self.file_path):
            return pd.read_csv(self.file_path)
        
        self.sampling_params = GenerationConfig(**self.cfg.generation_config)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_id).to(self.device)
        
        data = load_dataset(self.hf_path, split='test').to_pandas()
        data = self.truncate_context(data) 
        data = self.generate_answers(data)
        data = self.gpt_correctness(data)
        data.to_csv(self.file_path, index=False)
        return data
        
    def truncate_context(self, data):
        """
        This function needs to truncate context length. In transformer lens the max prompt length is 2048.
        """
        max_tokens = 2000
    
        def truncate(row):
            question = row['question']
            context = row['context']
    
            while True:
                messages = [
                    {"role": "user", "content": w_context_user_prompt(question, context)},
                    {"role": "system", "content": w_context_system_prompt()}
                ]
                question_template = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                tokenized = self.tokenizer(
                    question_template, return_tensors="pt", add_special_tokens=False
                )
                total_tokens = tokenized["input_ids"].shape[1]
    
                if total_tokens <= max_tokens:
                    break
    
                context_tokens = self.tokenizer(
                    context, return_tensors="pt", add_special_tokens=False
                )["input_ids"][0]
    
                context_tokens = context_tokens[:-100]
                context = self.tokenizer.decode(context_tokens, skip_special_tokens=True)
                
            return context
    
        data['context'] = data.apply(truncate, axis=1)
        return data

    def gpt_correctness(self, df):
        for index, row in tqdm(df.iterrows()):

            is_correct_wo_context = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": correctness_user_prompt(
                        question=row['question'], 
                        golden_answer=row['reference'], 
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
                        golden_answer=row['reference'], 
                        model_answer=row['our_answer_w_context']
                    )},
                    {"role": "system", "content": correctness_system_prompt()}
                ]
            ).choices[0].message.content

            df.loc[index, 'is_correct_wo_context'] = is_correct_wo_context
            df.loc[index, 'is_correct_w_context'] = is_correct_w_context

        return df
    
    def generate_answers(self, df):
        try:
            max_new_tokens = int(df['reference'].apply(len).mean())
        except:
            # if reference is not provided
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
                **input_data,
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


class NQDataset(AbstractDataset):
    def __init__(self):
        super().__init__(
            hf_path='aboriskin/adaptive_rag_nq',
            file_path='data/adaptive_rag_natural_questions_test.csv'
        )


class WikiMultiHopQADataset(AbstractDataset):
    def __init__(self):
        super().__init__(
            hf_path='aboriskin/adaptive_rag_2wikimultihopqa',
            file_path='data/adaptive_rag_2wikimultihopqa_test.csv'
        )


class HotPotQADataset(AbstractDataset):
    def __init__(self):
        super().__init__(
            hf_path='aboriskin/adaptive_rag_hotpotqa',
            file_path='data/adaptive_rag_hotpotqa_test.csv'
        )


class MusiqueDataset(AbstractDataset):
    def __init__(self):
        super().__init__(
            hf_path='aboriskin/adaptive_rag_musique',
            file_path='data/adaptive_rag_musique_test.csv'
        )
