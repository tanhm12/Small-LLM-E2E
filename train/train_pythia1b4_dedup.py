import os

### some tweaks to make bitandbytes run
os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["PATH"] = "${CUDA_HOME}/bin:${PATH}"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  ## specify your gpu number

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling


tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b-deduped", cache_dir="./models")
tokenizer.pad_token = "<|padding|>"
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1.4b-deduped", device_map='auto', torch_dtype=torch.float16, cache_dir="./models")



from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

def get_lora_model(model):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.05
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

model = get_lora_model(model)


############### load dataset  #####################
from datasets import Dataset
import json 

def cut_conversations_to_even_turns(conversation):
    last = len(conversation) if len(conversation) % 2 == 0 else len(conversation) - 1
    return conversation[:last]

with open("data/oassten_dolly.json") as f:
    combined_data_json = json.load(f)
combined_data_json = list(map(cut_conversations_to_even_turns, combined_data_json))  ## 34126
combined_data_json = [item for item in combined_data_json if len(item) > 1]
splitted_combine_data_json = combined_data_json

ds = Dataset.from_dict({"id": list(range(len(splitted_combine_data_json))), "conversation": splitted_combine_data_json})  


import random
from dataclasses import dataclass
from typing import Dict, Sequence
import numpy as np

IGNORE_INDEX = -100
# prompt_format = "### Human: {human}### Assistant: {bot}" + tokenizer.eos_token  ## alpaca format 
prompt_format = "Human:\n{human}" + tokenizer.eos_token + "\nAssistant:\n{bot}" + tokenizer.eos_token

TURN_CONCATENATION_TOKEN = ""

@dataclass
class DataCollatorForMultiTurnsConversation(object):  ## DataCollatorForSeq2Seq
    tokenizer: transformers.PreTrainedTokenizer
    max_length: int
    label_pad_token_id: int = IGNORE_INDEX

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        data = []
        for example in instances:
            text = []
            conversation = example['conversation']
            for i in range(0, len(conversation), 2):
                text.append(prompt_format.format(**{
                    "human": conversation[i]["text"].strip(),
                    "bot": conversation[i+1]["text"].strip()
                }))
            data.append(TURN_CONCATENATION_TOKEN.join(text))
        if random.random() < 0.001:
            print(data)
        tokenized_conversations = self.tokenizer(
            data,
            max_length=self.max_length,
            truncation=True
        )
        
        labels = [] 
        batch_max_len = max(len(item) for item in tokenized_conversations['input_ids'])
        padding_side = self.tokenizer.padding_side
        
        for tokenized_source in tokenized_conversations['input_ids']:
            remainder = [self.label_pad_token_id] * (batch_max_len - len(tokenized_source))
            if isinstance(tokenized_source, list):
                label = tokenized_source + remainder if padding_side == "right" else remainder + tokenized_source
            elif padding_side == "right":
                label = np.concatenate([tokenized_source, remainder]).astype(np.int64)
            else:
                label = np.concatenate([remainder, tokenized_source]).astype(np.int64)
            labels.append(torch.tensor(label))  ## https://github.com/huggingface/transformers/blob/v4.29.1/src/transformers/models/bloom/modeling_bloom.py#L827
        
        tokenized = self.tokenizer.pad(tokenized_conversations, max_length=self.max_length, return_tensors="pt", verbose=False)
        data_dict = {
            'input_ids': tokenized.input_ids.long(),
            'attention_mask': tokenized.attention_mask.long(),
        }
        data_dict['labels'] = torch.stack(labels).long()
        return data_dict


    
import os
import sys

import torch
import transformers

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)



def train(
    # model/data params
    model: AutoModelForCausalLM,
    tokenizer,
    train_ds: Dataset,
    val_ds: Dataset=None,
    output_dir: str = "./save/pythia1b4-chat-oasst-dolly",
    # training hyperparams
    batch_size: int = 24,
    micro_batch_size: int = 3,
    num_epochs: int = 3,
    learning_rate: float = 8e-5,
    cutoff_len: int = 640,
    val_set_ratio: float = 0.05,
    warmup_steps=400,
    logging_steps=100,
    eval_steps=1340,
    save_steps=1340,
    save_total_limit=3,
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {model.config._name_or_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_ratio: {val_set_ratio}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
        )
        
    gradient_accumulation_steps = batch_size // micro_batch_size

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    
    if val_ds is None:
        if val_set_ratio > 0:
            train_ds = train_ds.train_test_split(
                test_size=val_set_ratio, shuffle=True, seed=22
            )
            val_ds = train_ds["test"].shuffle()
            train_ds = train_ds["train"].shuffle()
        else:
            train_ds = train_ds.shuffle()
            val_ds = None
    else:
        train_ds = train_ds.shuffle()
        val_ds = val_ds.shuffle()

    print(len(train_ds), len(val_ds))

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            eval_accumulation_steps=1,
            do_train = True,
            do_eval = True,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=logging_steps,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_ratio > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_steps if val_set_ratio > 0 else None,
            save_steps=save_steps,
            output_dir=output_dir,
            save_total_limit=save_total_limit,
            remove_unused_columns=False,
            report_to= None,
            # run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=DataCollatorForMultiTurnsConversation(
            tokenizer, cutoff_len
        ),
    )
    model.config.use_cache = False

    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)


train(model, tokenizer, ds )