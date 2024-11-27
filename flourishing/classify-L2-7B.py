# Classify tweets using LLAMA2-7B 
# (S.M.Iacus 2024)

from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import pandas as pd

mycsv = 'classified-L2-7B.csv'


template1 = """Classify this text according to the well-being dimensions listed below.

Text: {text}

Use only the scale: 'low', 'medium' and 'high'. Return a JSON dictionary that contains only the well-being dimensions that apply. Do not explain your reasoning. Only classify those categories you are most confident of and report the probability of each using this template: 'well-being dimension, scale value, probability value'

Well-being dimensions:
[Happiness, Resilience, Self-esteem, Life satisfaction, Fear of future, Vitality, Having energy, Positive functioning, Expressing job satisfaction, Expressing optimism, Peace with thoughts and feelings, Purpose in life, Depression, Anxiety, Suffering, Feeling pain]. Answer:"""



template2 = """Classify this text according to the well-being dimensions listed below.


Text: {text}

Use only the scale: 'low', 'medium' and 'high'. Return a JSON dictionary that contains only the well-being dimensions that apply. Do not explain your reasoning. Only classify those categories you are most confident of and report the probability of each using this template: 'well-being dimension, scale value, probability value'

Well-being dimensions:
[Expressing altruism, Loneliness, Quality of relationships, Belonging to society, Expressing gratitude, Expressing trust, Feeling trusted, Balance in the various aspects of own life, Mastery (ability or capability), Perceiving discrimination, Feeling loved by God, Belief in God, Religious criticism, Spiritual punishment, Feeling religious comfort]. Answer:"""


template3 = """Classify this text according to the well-being dimensions listed below.

Text: {text}

Use only the scale: 'low', 'medium' and 'high'. Return a JSON dictionary that contains only the well-being dimensions that apply. Do not explain your reasoning. Only classify those categories you are most confident of and report the probability of each using this template: 'well-being dimension, scale value, probability value'

Well-being dimensions:
[Financial/material worry, Life after death belief, Volunteering, Charitable giving/helping, Seeking for forgiveness, Feeling having a political voice, Expressing government approval, Having hope, Promoting good, Expressing delayed gratification]. Answer:"""


template4 = """Classify this text according to the well-being dimensions listed below.

Text: {text}

Use only the scale: 'low', 'medium' and 'high'. Return a JSON dictionary that contains only the well-being dimensions that apply. Do not explain your reasoning. Only classify those categories you are most confident of and report the probability of each using this template: 'well-being dimension, scale value, probability value'

Well-being dimensions:
[PTSD (Post-traumatic stress disorder), Describing smoking related health issues, Describing drinking related health issues, Describing health limitations, Expressing empathy]. Answer:"""



prompt1 = PromptTemplate(template=template1, input_variables=["text"])
prompt2 = PromptTemplate(template=template2, input_variables=["text"])
prompt3 = PromptTemplate(template=template3, input_variables=["text"])
prompt4 = PromptTemplate(template=template4, input_variables=["text"])

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])




n_gpu_layers = 1  # Change this value based on your model and your GPU VRAM pool.
                  # -1 to offload everything to GPU
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Make sure the model path is correct for your system!
# Make sure to download a copy of llama-2-7b-chat.Q4_K_M.gguf from Huggingface. 
# We use the version in https://huggingface.co/TheBloke/Llama-2-7B-GGUF
llm = LlamaCpp(
    model_path="llama-2-7b-chat.Q4_K_M.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    f16_kv=True,  # MUST set to True
    callback_manager=callback_manager,
    verbose=True, 
    temperature=0.01,
    n_gqa=8,  
)

chain1 = LLMChain(prompt=prompt1, llm=llm)
chain2 = LLMChain(prompt=prompt2, llm=llm)
chain3 = LLMChain(prompt=prompt3, llm=llm)
chain4 = LLMChain(prompt=prompt4, llm=llm)


def toDF(s, df):
    s = s.lower()
    s = s.replace("{", "").replace("}", "").replace(":", ",").replace("\"", "").split("\n")
    for i in s:
        try:
            v = [i.split(",")[0].strip('\''), i.split(",")[1].strip('\'').replace(" ", ""), i.split(",")[2].strip('\'').replace(" ", "")]
        except:
            next
        else:
            df.loc[len(df)] = v 

def analyze(thistext, id):
    answer1 = chain1.run(thistext)
    answer2 = chain2.run(thistext)
    answer3 = chain3.run(thistext)
    answer4 = chain4.run(thistext)
 #   
    answer1 = answer1[answer1.find('{'):]
    answer1 = answer1[:answer1.find('}')+1]
    answer2 = answer2[answer2.find('{'):]
    answer2 = answer2[:answer2.find('}')+1]
    answer3 = answer3[answer3.find('{'):]
    answer3 = answer3[:answer3.find('}')+1]
    answer4 = answer4[answer4.find('{'):]
    answer4 = answer4[:answer4.find('}')+1]
    locdf = pd.DataFrame(columns=['dimension', 'value', 'confidence'])       
    toDF(answer1, locdf)
    toDF(answer2, locdf)
    toDF(answer3, locdf)
    toDF(answer4, locdf)
    locdf['id'] = id
    return(locdf)

data = pd.read_csv('sample.csv')

first = analyze(data.tweet_text[0], data.message_id[0])

first.to_csv(mycsv, mode='a', header=True, index=False)

for i in range(1,len(data)):
    id = data.loc[i, "message_id"]
    tweet = data.loc[i, "tweet_text"]
    print(f"\n\nN = {i}, id = {id}\ntweet = {tweet}\n")
    try:
        tmp = analyze(tweet, id)
    except:
        next
    else:
        print(tmp)
        tmp.to_csv(mycsv, mode='a', header=False, index=False)
        

exit(0)



 