# Tweets classifier using LLAMA-3.2-3B
# (S.M.Iacus 2024)

output_csv = 'classified-L32-3B.csv'


import os
import torch
from datasets import load_dataset


from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)



import pandas as pd


finetuned_model_path = "Llama-32-3B-Q4_K_M.gguf"

# the next one is used only to specify the tokenizer
finetuned_model = "siacus/Llama-32-3B-tweets-10-adapt"

compute_dtype = getattr(torch, "float16")

torch.device("mps")  # check

n_gpu_layers = -1  # should all go to GPU : Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path=finetuned_model_path,
    n_ctx=512,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    f16_kv=True,  # MUST set to True
    verbose=True, 
    temperature=0.01,
    streaming=False,
)

tokenizer = AutoTokenizer.from_pretrained(finetuned_model, trust_remote_code=True)

dims1 ="""[Happiness, Resilience, Self-esteem, Life satisfaction, Fear of future, Vitality, Having energy, Positive functioning, Expressing job satisfaction, Expressing optimism, Peace with thoughts and feelings, Purpose in life, Depression, Anxiety, Suffering, Feeling pain]"""

dims2 ="""[Expressing altruism, Loneliness, Quality of relationships, Belonging to society, Expressing gratitude, Expressing trust, Feeling trusted, Balance in the various aspects of own life, Mastery (ability or capability), Perceiving discrimination, Feeling loved by God, Belief in God, Religious criticism, Spiritual punishment, Feeling religious comfort]"""

dims3 ="""[Financial or material worry, Life after death belief, Volunteering, Charitable giving/helping, Seeking for forgiveness, Feeling having a political voice, Expressing government approval, Having hope, Promoting good, Expressing delayed gratification]"""

dims4 ="""[PTSD (Post-traumatic stress disorder), Describing smoking related health issues, Describing drinking related health issues, Describing health limitations, Expressing empathy]"""


instruction1 = f"""Please classify the following text based on the well-being dimensions listed below. Use only the scale: 'low', 'medium', and 'high'. Return a JSON dictionary that contains only the well-being dimensions that apply. Do not explain your reasoning. \n
Well-being dimensions:\n
{dims1}. """

instruction2 = f"""Please classify the following text based on the well-being dimensions listed below. Use only the scale: 'low', 'medium', and 'high'. Return a JSON dictionary that contains only the well-being dimensions that apply. Do not explain your reasoning. \n
Well-being dimensions:\n
{dims2}. """

instruction3 = f"""Please classify the following text based on the well-being dimensions listed below. Use only the scale: 'low', 'medium', and 'high'. Return a JSON dictionary that contains only the well-being dimensions that apply. Do not explain your reasoning. \n
Well-being dimensions:\n
{dims3}. """

instruction4 = f"""Please classify the following text based on the well-being dimensions listed below. Use only the scale: 'low', 'medium', and 'high'. Return a JSON dictionary that contains only the well-being dimensions that apply. Do not explain your reasoning. \n
Well-being dimensions:\n
{dims4}. """


template1 = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|> {instruction1}.
    Here is the text: \" {{text}} \".<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

template2 = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|> {instruction2}.
    Here is the text: \" {{text}} \".<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

template3 = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|> {instruction3}.
    Here is the text: \" {{text}} \".<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

template4 = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|> {instruction4}.
    Here is the text: \" {{text}} \".<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


prompt1 = PromptTemplate(template=template1, input_variables=["text"])
prompt2 = PromptTemplate(template=template2, input_variables=["text"])
prompt3 = PromptTemplate(template=template3, input_variables=["text"])
prompt4 = PromptTemplate(template=template4, input_variables=["text"])


chain1 = LLMChain(prompt=prompt1, llm=llm)
chain2 = LLMChain(prompt=prompt2, llm=llm)
chain3 = LLMChain(prompt=prompt3, llm=llm)
chain4 = LLMChain(prompt=prompt4, llm=llm)

def toDF(s, df):
    s = s.lower()
    s = s.replace("{", "").replace("}", "").replace("\n", "").split(",")   
    for i in s:
        try:
            v = [i.split(":")[0].strip(' ').strip('\'').replace("\"", ""), i.split(":")[1].strip(' ').strip('\'').replace("\"", "").replace(" ", "")]
        except:
            next
        else:
            df.loc[len(df)] = v 


def analyze(thistext, id):
    answer1 = chain1.invoke(thistext)['text']
    answer2 = chain2.invoke(thistext)['text']
    answer3 = chain3.invoke(thistext)['text']
    answer4 = chain4.invoke(thistext)['text']
    answer1 = answer1[answer1.find('{'):]
    answer1 = answer1[:answer1.find('}')+1]
    answer2 = answer2[answer2.find('{'):]
    answer2 = answer2[:answer2.find('}')+1]
    answer3 = answer3[answer3.find('{'):]
    answer3 = answer3[:answer3.find('}')+1]
    answer4 = answer4[answer4.find('{'):]
    answer4 = answer4[:answer4.find('}')+1]
    locdf = pd.DataFrame(columns=['dimension', 'value'])       
    toDF(answer1, locdf)
    toDF(answer2, locdf)
    toDF(answer3, locdf)
    toDF(answer4, locdf)
    locdf['id'] = id
    return(locdf)



tweets = pd.read_csv("sample.csv")
tweets = tweets.rename({'message_id': 'id', 'tweet_text' : 'tweet'}, axis=1)

data = tweets.reset_index(drop=True)


first = analyze(data.tweet[0], data.id[0])

first.to_csv(output_csv, mode='a', header=True, index=False)

n = len(data)

for i in range(1,len(data)):
    id = data.loc[i, "id"]
    tweet = data.loc[i, "tweet"]
    print(f"\n\nN = {i}/{n}, id = {id}\ntweet = {tweet}\n")
    try:
        tmp = analyze(tweet, id)
    except:
        next
    else:
        print(tmp)
        tmp.to_csv(output_csv, mode='a', header=False, index=False)
        

exit(0)


