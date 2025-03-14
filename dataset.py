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
        # if os.path.exists(self.file_path):
        #     return pd.read_csv(self.file_path)
        
        self.generation_config = GenerationConfig(**self.cfg.generation_config)
        self.dola_generation_config = GenerationConfig(**self.cfg.dola_generation_config)
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_id).to(self.device)
        
        data = load_dataset(self.hf_path, split='train').to_pandas() # only one split
        data = self.generate_answers(data)
        data = self.generate_dola_answers(data)
        # data = self.gpt_correctness(data)
        data.to_csv(self.file_path, index=False)
        return data
        
    # def gpt_correctness(self, df):
    #     for index, row in tqdm(df.iterrows()):

    #         is_correct_wo_context = self.client.chat.completions.create(
    #             model="gpt-4o",
    #             messages=[
    #                 {"role": "user", "content": correctness_user_prompt(
    #                     question=row['question'], 
    #                     golden_answer=row['reference'], 
    #                     model_answer=row['our_answer_wo_context']
    #                 )},
    #                 {"role": "system", "content": correctness_system_prompt()}
    #             ]
    #         ).choices[0].message.content

    #         is_correct_w_context = self.client.chat.completions.create(
    #             model="gpt-4o",
    #             messages=[
    #                 {"role": "user", "content": correctness_user_prompt(
    #                     question=row['question'], 
    #                     golden_answer=row['reference'], 
    #                     model_answer=row['our_answer_w_context']
    #                 )},
    #                 {"role": "system", "content": correctness_system_prompt()}
    #             ]
    #         ).choices[0].message.content
            
    #         is_correct_dola = self.client.chat.completions.create(
    #             model="gpt-4o",
    #             messages=[
    #                 {"role": "user", "content": correctness_user_prompt(
    #                     question=row['question'], 
    #                     golden_answer=row['reference'], 
    #                     model_answer=row['our_answer_dola']
    #                 )},
    #                 {"role": "system", "content": correctness_system_prompt()}
    #             ]
    #         ).choices[0].message.content

    #         df.loc[index, 'is_correct_wo_context'] = is_correct_wo_context
    #         df.loc[index, 'is_correct_w_context'] = is_correct_w_context
    #         df.loc[index, 'is_correct_dola'] = is_correct_dola

    #     return df
    
    def calculate_max_new_tokens(self, df):
        try:
            reference_tokens = self.tokenizer(
                df['reference'].tolist(), 
                padding=False, 
                truncation=False, 
                return_tensors=None
            )
            avg_token_count = sum(len(ids) for ids in reference_tokens['input_ids']) / len(reference_tokens['input_ids'])
            max_new_tokens = int(avg_token_count * 1.2)
        except:
            max_new_tokens = 100

        return max_new_tokens
    
    # def clean_generated_text(self, text):
    #     # """Очищает сгенерированный текст от артефактов"""
    #     # # Удаляем префиксы ассистента, если они есть
    #     # if "assistant" in text.lower():
    #     #     text = text.split("assistant", 1)[-1].strip()
        
    #     # # Удаляем начальные двоеточия или другие артефакты
    #     # text = text.lstrip(": ")
        
    #     return text
    
    def generate_answers(self, df):
        max_new_tokens = 20 # self.calculate_max_new_tokens(df)
        
        #########################################################
        ############## 1. Prepare inputs ########################
        #########################################################

        wo_context_inputs = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing wo_context inputs"):
            messages = [
                {"role": "system", "content": wo_context_system_prompt()},
                {"role": "user", "content": row['question']}
            ]
            input_str = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            wo_context_inputs.append(input_str)
        
        w_context_inputs = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing w_context inputs"):
            messages = [
                {"role": "system", "content": w_context_system_prompt()},
                {"role": "user", "content": w_context_user_prompt(row['question'], row['context'])}
            ]
            input_str = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            w_context_inputs.append(input_str)
        
        #########################################################
        ############## 2. Generate answers ######################
        #########################################################

        wo_context_answers = []
        for i in tqdm(range(0, len(wo_context_inputs), self.cfg.batch_size), desc="Generating wo_context answers"):
            batch_inputs = wo_context_inputs[i:i+self.cfg.batch_size]
            tokenized_inputs = self.tokenizer(
                batch_inputs,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
                pad_to_multiple_of=8
            ).to(self.device)
            
            input_lens = tokenized_inputs["attention_mask"].sum(dim=1)
            
            outputs = self.model.generate(
                **tokenized_inputs,
                max_new_tokens=max_new_tokens,
                generation_config=self.generation_config,
            )
            
            batch_answers = [
                self.tokenizer.decode(
                    output[input_len:], 
                    skip_special_tokens=True
                )
                for output, input_len in zip(outputs, input_lens)
            ]
            wo_context_answers.extend(batch_answers)
        
        w_context_answers = []
        for i in tqdm(range(0, len(w_context_inputs), self.cfg.batch_size), desc="Generating w_context answers"):
            batch_inputs = w_context_inputs[i:i+self.cfg.batch_size]
            tokenized_inputs = self.tokenizer(
                batch_inputs,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
                pad_to_multiple_of=8
            ).to(self.device)
            
            input_lens = tokenized_inputs["attention_mask"].sum(dim=1)
            
            outputs = self.model.generate(
                **tokenized_inputs,
                max_new_tokens=max_new_tokens,
                generation_config=self.generation_config,
            )
            
            batch_answers = [
                self.tokenizer.decode(
                    output[input_len:], 
                    skip_special_tokens=True
                )
                for output, input_len in zip(outputs, input_lens)
            ]
            w_context_answers.extend(batch_answers)
        
        df['our_answer_wo_context'] = wo_context_answers
        df['our_answer_w_context'] = w_context_answers
        
        return df
    
    def generate_dola_answers(self, df):
        max_new_tokens = self.calculate_max_new_tokens(df)
            
        dola_inputs = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing DOLA inputs"):
            messages = [
                {"role": "system", "content": wo_context_system_prompt()},
                {"role": "user", "content": row['question']}
            ]
            input_str = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            dola_inputs.append(input_str)
        
        dola_answers = []
        for i in tqdm(range(0, len(dola_inputs), self.cfg.batch_size), desc="Generating DOLA answers"):
            batch_inputs = dola_inputs[i:i+self.cfg.batch_size]
            tokenized_inputs = self.tokenizer(
                batch_inputs,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
                pad_to_multiple_of=8
            ).to(self.device)
            
            input_lens = tokenized_inputs["attention_mask"].sum(dim=1)
            
            outputs = self.model.generate(
                **tokenized_inputs,
                max_new_tokens=max_new_tokens,
                generation_config=self.dola_generation_config,
            )
            
            batch_answers = [
                self.tokenizer.decode(
                        output[input_len:], 
                        skip_special_tokens=True
                    )
                for output, input_len in zip(outputs, input_lens)
            ]
            dola_answers.extend(batch_answers)
        
        df['our_answer_dola'] = dola_answers
        
        return df


class NQDataset(AbstractDataset):
    def __init__(self):
        super().__init__(
            hf_path='aboriskin/adaptive_rag_nq',
            file_path='data/adaptive_rag_natural_questions.csv'
        )


class WikiMultiHopQADataset(AbstractDataset):
    def __init__(self):
        super().__init__(
            hf_path='aboriskin/adaptive_rag_2wikimultihopqa',
            file_path='data/adaptive_rag_2wikimultihopqa.csv'
        )


class HotPotQADataset(AbstractDataset):
    def __init__(self):
        super().__init__(
            hf_path='aboriskin/adaptive_rag_hotpotqa',
            file_path='data/adaptive_rag_hotpotqa.csv'
        )


class MusiqueDataset(AbstractDataset):
    def __init__(self):
        super().__init__(
            hf_path='aboriskin/adaptive_rag_musique',
            file_path='data/adaptive_rag_musique.csv'
        )
