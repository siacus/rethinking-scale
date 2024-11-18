# tests performance of fine-tuning 
# script v2
# (S.M.Iacus 2024)


from llama_cpp import Llama
from pprint import pprint
from datasets import load_dataset
import re
import pandas as pd
from tqdm import tqdm

# Create an instance of Llama to load the model
llm = Llama(
    model_path="llama-2-7b-cap_verified-Q4_K_M.gguf",
    n_ctx = 2048,
    n_gpu_layers = -1,
)

print("Model loaded")
cap_dataset = "siacus/cap_pe_verified"
dataset = load_dataset(cap_dataset, split={'train': 'train', 'test': 'test'})


def get_response(answer):
    match = re.search(r'Answer =.*?(\d+)', answer)
    if match:
        return int(match.group(1))
    else:
        return(0)

df = pd.DataFrame(dataset['train'])

n = df.shape[0]
cnt = 0
results = []
for index, row in tqdm(df.iterrows(), total=n, desc="Processing rows"):
#for index, row in df.iterrows():
    truth = row['macroCode']
    idCoding = row['idCoding']
    idOrig = row['idOrig']
    query = row['text']
    query_start = query.find('Answer =')
    query = query[:query_start + len('Answer =')].strip()
    output = llm(
        query,  # Prompt
        max_tokens=50,  # Generate up to 128 tokens
        echo=True,  # Echo the prompt back in the output
        temperature = 0.01
    )
    answer = output['choices'][0]['text']
    response = get_response(answer)
    isCorrect = (truth == response)
    results.append({
        'idCoding': idCoding,
        'idOrig': idOrig,
        'trueCat': truth,
        'predicted': response,
        'isCorrect' : isCorrect
    })
    cnt = cnt + isCorrect
    print(f"num True = {cnt} out of {len(results)} [of {n}]")


results_df = pd.DataFrame(results)
prop = results_df['isCorrect'].value_counts(normalize=True).sort_index()
print(prop)

results_df.to_csv('classification_train-L2-7B-cap-verified-v2.csv', index=False)

df = pd.DataFrame(dataset['test'])


n = df.shape[0]
cnt = 0
results = []
for index, row in tqdm(df.iterrows(), total=n, desc="Processing rows"):
#for index, row in df.iterrows():
    truth = row['macroCode']
    idCoding = row['idCoding']
    idOrig = row['idOrig']
    query = row['text']
    query_start = query.find('Answer =')
    query = query[:query_start + len('Answer =')].strip()
    output = llm(
        query,  # Prompt
        max_tokens=50,  # Generate up to 128 tokens
        echo=True,  # Echo the prompt back in the output
        temperature = 0.01
    )
    answer = output['choices'][0]['text']
    response = get_response(answer)
    isCorrect = (truth == response)
    results.append({
        'idCoding': idCoding,
        'idOrig': idOrig,
        'trueCat': truth,
        'predicted': response,
        'isCorrect' : isCorrect
    })
    cnt = cnt + isCorrect
    print(f"num True = {cnt} out of {len(results)} [of {n}]")


results_df = pd.DataFrame(results)
prop = results_df['isCorrect'].value_counts(normalize=True).sort_index()
print(prop)

results_df.to_csv('classification_test-L2-7B-cap-verified-v2.csv', index=False)


