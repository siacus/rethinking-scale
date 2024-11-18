# This script uses a large subset of DV data to fine-tune LLAMA2
# (S.M.Iacus 2024)

from huggingface_hub import HfApi, HfFolder
#HfFolder.save_token("hf_YOUR_HF_TOKEN") # only need to to once
api = HfApi()
api.whoami()  # This should print your user info



from huggingface_hub import HfApi, HfFolder
#HfFolder.save_token("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")
api = HfApi()
api.whoami()  # This should print your user info


import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, pipeline
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import wandb
from accelerate import Accelerator
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

# Initialize the accelerator
accelerator = Accelerator() #device_placement=True)

print("Number of GPUs:", torch.cuda.device_count())

# Define hyperparameters
learning_rate = 2e-4
gradient_accumulation_steps = 1
per_device_train_batch_size = 32
num_train_epochs = 10
max_seq_length = 2048
max_steps = -1
optimizer = "paged_adamw_32bit"
max_grad_norm = 0.3
max_length = 2048

# Initialize W&B with a custom run name
wandb.init(
    project="dv",
    reinit=True,
    name="llama-2-7b-dv",
    config={
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_seq_length": max_seq_length,
        "max_steps": max_steps,
        "optimizer": optimizer,
        "max_grad_norm": max_grad_norm,
    }
)




# Model and dataset
base_model = "meta-llama/Llama-2-7b-chat-hf" 
cap_dataset = "siacus/dv_subject" # this data are prepared for training the model
new_model = "siacus/llama-2-7b-dv" # you need to change 'siacus' to your HF account

dataset = load_dataset(cap_dataset, split={'train': 'train', 'test': 'test'})

# Quantization configuration
quant_config = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.float16,
    "bnb_4bit_use_double_quant": False,
}

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(base_model, 
    quantization_config=quant_config, 
    device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model.config.use_cache = False # only for training
model.config.pretraining_tp = 1

# Use Accelerator to handle device placement
model, tokenizer = accelerator.prepare(model, tokenizer)

# PEFT configuration
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1, # like lasso
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

mp = get_peft_model(model, peft_params)
mp.print_trainable_parameters()

# Training arguments
training_params = TrainingArguments(
    output_dir="./resultsdv",
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    optim=optimizer,
    weight_decay=0.001,
    save_steps=25,
    logging_steps=25,
    fp16=False,
    gradient_checkpointing=True,
    bf16=False,
    max_steps=max_steps,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="wandb",
    max_grad_norm=max_grad_norm,
    dataloader_num_workers = 4,
    dataloader_pin_memory = True
)



# Initialize the Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

# Prepare the trainer with Accelerator
trainer.accelerator = accelerator

trainer.train()


# Save the trained model
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)


# Push to hub under your namespace
trainer.model.push_to_hub(new_model)
trainer.tokenizer.push_to_hub(new_model)

# Finish W&B run
wandb.finish()
