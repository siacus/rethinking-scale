# This script is used to mere the weights after fine-tuning
# (S.M.Iacus 2024)
# This script requires enough memory to load the models
# The example works for the "llama-2-7b-small-dv" model but
# all the other models have been merged similarly

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
    BitsAndBytesConfig
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import torch

base_model = "meta-llama/Llama-2-7b-chat-hf" 
hf_user = 'siacus' # replace with your HF account
new_model_name = "llama-2-7b-small-dv" # replace with the one you like
new_model = hf_user + '/' + new_model_name

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map={"": 0},  # or specify differently
)

# here we save the adpaters, just in case
finetuned_model = PeftModel.from_pretrained(model, new_model)
finetuned_model.save_pretrained(hf_user + '/' + "adapters" + '/' + new_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.save_pretrained(hf_user + '/' + "adapters" + '/' + new_model_name)

# here we merge adapters and original model weights
merged_model = finetuned_model.merge_and_unload()
merged_model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)

