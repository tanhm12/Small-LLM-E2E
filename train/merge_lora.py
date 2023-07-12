import os

os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["PATH"] = "${CUDA_HOME}/bin:${PATH}"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import torch
import os
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, set_peft_model_state_dict, LoraConfig



tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b-deduped", cache_dir="models")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1.4b-deduped", device_map='auto', cache_dir="models",
                                             torch_dtype=torch.float16, 
                                             )

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
peft_config = LoraConfig(
    # task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.05
)


model = get_peft_model(model, peft_config)
checkpoint_name = os.path.join("save/pythia1b4-chat-oasst-dolly/checkpoint-4020", "adapter_model.bin")
adapters_weights = torch.load(checkpoint_name, 
                              map_location=torch.device('cpu')
                              )
set_peft_model_state_dict(model, adapters_weights)



model = model.merge_and_unload()

save_dir = "models/pythia1b4-chat-oasst-dolly"
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)