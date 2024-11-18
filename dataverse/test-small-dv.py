# tests performance of fine-tuning 
# (S.M.Iacus 2024)

from llama_cpp import Llama
from pprint import pprint
from datasets import load_dataset
import re
import json
import pandas as pd
from tqdm import tqdm


base_model = "meta-llama/Llama-2-7b-chat-hf" 
hf_user = 'siacus' # replace with your HF account
new_model_name = "llama-2-7b-small-dv" # replace with the one you used for fine-tuning
new_model = hf_user + '/' + new_model_name

ctx_window = 2048 # in on-line we should probably make it adaptive
# Create an instance of Llama to load the model
# model_path - The model we want to load
model = Llama(
    model_path= hf_user + '/' + new_model_name + "-Q4_K_M.gguf",  
    n_ctx = ctx_window,
    # device='metal'  # change to your architecture to accelerate inference or drop it
)


dv_dataset = "siacus/dv_subject"


dataset = load_dataset(dv_dataset, split={'train': 'train', 'test': 'test'})

def get_responses(answer):
    answer_start = answer.find('text labels:')
    if answer_start != -1:
        answer_text = answer[answer_start + len('text labels:'):].strip()
        pattern = r"\[([^\]]+)\]"
        matches = re.search(pattern, answer_text)
        if matches:
            labels_str = matches.group(1)
            labels_list = [label.strip().strip("'") for label in labels_str.split("', '") if label.strip()]
            return labels_list[:3]
    return []


df = pd.DataFrame(dataset['test'])


n = df.shape[0]
cnt = 0
results = []
for index, row in tqdm(df.iterrows(), total=n, desc="Processing rows"):
    query = row['prompt'][:ctx_window] # we trim the input
    truth = row['Subject']
    dvid = row['id']
    doi = row['doi']
    try: # we need to catch errors for too long queries.
        output = model(
            query,  # Prompt
            max_tokens=50,  # Generate up to 50 tokens
            echo=True,  # Echo the prompt back in the output
            temperature=0.01
        )
        answer = output['choices'][0]['text']
        responses = get_responses(answer)
    except Exception as e:
        print(f"Error processing row {index}: {str(e)}")
        responses = []  # Define a default response in case of error
    # Check if any element in truth exists in responses
    if any(item in responses for item in truth):
        isCorrect = True
    else:
        isCorrect = False   
    results.append({
        'id': dvid,
        'doi': doi,
        'struSubject': truth,
        'predicted': responses,
        'isCorrect' : isCorrect
    })
    cnt = cnt + isCorrect
    print(f"num True = {cnt} out of {len(results)} [of {n}]")
    results_df = pd.DataFrame(results)
    prop = results_df['isCorrect'].value_counts(normalize=True).sort_index()
    print(prop)



results_df = pd.DataFrame(results)
prop = results_df['isCorrect'].value_counts(normalize=True).sort_index()
print(prop)

results_df.to_csv('classification_results-' + new_model_name + '.csv', index=False)
