import os
from dotenv import load_dotenv

os.environ["CUDA_VISIBLE_DEVICES"] = "" 

load_dotenv()
HF_TOKEN = os.environ["HF_TOKEN"]

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling


tokenizer = AutoTokenizer.from_pretrained("models/pythia1b4-chat-oasst-dolly", cache_dir="./models")
model = AutoModelForCausalLM.from_pretrained("models/pythia1b4-chat-oasst-dolly", device_map='auto', torch_dtype=torch.float16, cache_dir="./models")


model.push_to_hub("Zayt/pythia1b4-chat-oasst-dolly", token=HF_TOKEN)
tokenizer.push_to_hub("Zayt/pythia1b4-chat-oasst-dolly", token=HF_TOKEN)