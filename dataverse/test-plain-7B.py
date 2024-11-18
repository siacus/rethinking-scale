# tests performance of fine-tuning (S.M.Iacus 2024)

# nohup python -u test-plain-7B.py > log-plain7B.txt &!

from llama_cpp import Llama
# We'll use pprint to more clearly look at the output
from pprint import pprint
from datasets import load_dataset
import re
import json
import pandas as pd
from tqdm import tqdm






#base_model = "TheBloke/llama-2-13b-chat.q4_K_M.gguf" #"meta-llama/Llama-2-7b-chat-hf" #"NousResearch/Llama-2-7b-chat-hf"
#hf_user = 'siacus' # replace with yours
#new_model_name = "llama-2-7b-small-dv" # replace with the one you like
#new_model = hf_user + '/' + new_model_name

ctx_window = 2048 # in on-line we should probably make it adaptive
# Create an instance of Llama to load the model
# model_path - The model we want to load
model = Llama(
    model_path = 'llama-2-7b-chat.q4_K_M.gguf',
#    model_path= hf_user + '/' + new_model_name + "-Q4_K_M.gguf",  
    n_ctx = ctx_window,
    n_gpu_layers = -1,
)




cap_dataset = "siacus/dv_subject"



dataset = load_dataset(cap_dataset, split={'train': 'train', 'test': 'test'})


aa = dataset['train']['Subject']
def get_unique_categories(aa):
    unique_categories = set()
    for sublist in aa:
        unique_categories.update(sublist)
    return list(unique_categories)

unique_categories = get_unique_categories(aa)
print(unique_categories)


def find_categories_in_text(text):
    found_categories = [category for category in unique_categories if category in text]
    return found_categories

def get_responses(answer):
    answer_start = answer.find('text labels:')
    if answer_start != -1:
        answer_text = answer[answer_start + len('text labels:'):].strip()
        matches = find_categories_in_text(answer_text)
        return(matches)
    return []



# try some id's
id = 10012
query = dataset['test']['prompt'][id]


max_tokens = 150

truth = dataset['test']['Subject'][id]
output = model(
        query,  # Prompt
        max_tokens=max_tokens,  # Generate up to 128 tokens
        echo=True,  # Echo the prompt back in the output
        temperature = 0.01
    )
answer = output['choices'][0]['text']
get_responses(answer) 
print(truth)

df = pd.DataFrame(dataset['test'])


n = df.shape[0]
cnt = 0
results = []
for index, row in tqdm(df.iterrows(), total=n, desc="Processing rows"):
#for index, row in df.iterrows():
    query = row['prompt'][:ctx_window] # we trim the input
    truth = row['Subject']
    dvid = row['id']
    doi = row['doi']
    try: # we need to catch errors for too long queries.
        output = model(
            query,  # Prompt
            max_tokens=max_tokens,  # Generate up to 50 tokens
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
        'trueSubject': truth,
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

results_df.to_csv('classification_results-7B-' + 'NO-FT' + '.csv', index=False)
