from abc import ABC
import os
import re

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
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_id).to(self.device)
        
        data = load_dataset(self.hf_path, split="train").to_pandas() # only one split
        data = self.generate_answers(data)
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

    def generate_answers(self, df):
        """
        Generates answers in three modes:
        1. Without context
        2. With context
        3. DOLA without context
        """
        generation_settings = {
            "wo_context": {
                "system_prompt_func": wo_context_system_prompt,
                "user_prompt_func": lambda q, c: q,  # ignore context
                "gen_config": self.generation_config,
                "out_column": "our_answer_wo_context"
            },
            "w_context": {
                "system_prompt_func": w_context_system_prompt,
                "user_prompt_func": lambda q, c: w_context_user_prompt(q, c),
                "gen_config": self.generation_config,
                "out_column": "our_answer_w_context"
            },
            "dola": {
                "system_prompt_func": wo_context_system_prompt,
                "user_prompt_func": lambda q, c: q,  # ignore context
                "gen_config": self.dola_generation_config,
                "out_column": "our_answer_dola"
            }
        }
        max_new_tokens = 20

        for setting_name, conf in generation_settings.items():
            
            system_prompt_func = conf["system_prompt_func"]
            user_prompt_func = conf["user_prompt_func"]
            gen_config = conf["gen_config"]
            out_column = conf["out_column"]
            batch_size = self.cfg.batch_size if setting_name != "w_context" else self.cfg.batch_size // 4

            prompts = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Preparing inputs"):
                system_text = system_prompt_func()
                user_text = user_prompt_func(row["question"], row.get("context", ""))
                
                messages = [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user_text}
                ]
                input_str = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                prompts.append(input_str)

            answers = []
            for i in tqdm(range(0, len(prompts), batch_size), desc=setting_name):
                batch_inputs = prompts[i : i + batch_size]
                
                tokenized_inputs = self.tokenizer(
                    batch_inputs,
                    padding=True,
                    return_tensors="pt",
                    add_special_tokens=False
                ).to(self.device)
                
                outputs = self.model.generate(
                    **tokenized_inputs,
                    max_new_tokens=max_new_tokens,
                    generation_config=gen_config,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                
                for j, out_ids in enumerate(outputs):
                    prompt_len_tokens = tokenized_inputs["input_ids"][j].shape[0]
                    gen_ids = out_ids[prompt_len_tokens:]
                    answer_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                    answers.append(answer_text)
            
            df[out_column] = answers
        
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
