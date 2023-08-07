# Requirements:
1. Python 3.10.6 or above.
2. requirements.txt
# Training
Using the famous Huggingface for training interface.
## Method
- Default model is `EleutherAI/pythia-1.4b-deduped`, which is based on GPT-NeoX, main language is English.
- Using LoRA on QKV matrices with 16bit precision of the base model.
- Supervised finetuning
## Data
- Mix of OASST1 (English subset) and [databrick/dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
- Preprocessed data at `data/oassten_dolly.json`, see `script.ipynb` for preprocessing pipeline 
## Parameters
These configs were tested on a single 11GB GPU (such as RTX 2080Ti).
```python
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
)
```
## Run the code
```bash
python train_pythia1b4_dedup.py
```
# Merge LoRA weights
After the training process, you can merge the trained LoRA weights into base model for easier use in the future, see `merge_lora.py`.
# Notes
For larger model, you might want to use 8 bit (or even 4 bit, but I prefer 8 bit) base model. It would reduce VRAM usage but also make the result slightly worse than 16 bit. 
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b-deduped", use_fast=False, cache_dir="./models")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-2.8b-deduped", device_map='auto', cache_dir="./models",
                                             quantization_config=bnb_config, 
                                             torch_dtype=torch.float16,
                                            #  load_in_8bit=True,
                                             )

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.05
)
model = get_peft_model(model, peft_config)
```