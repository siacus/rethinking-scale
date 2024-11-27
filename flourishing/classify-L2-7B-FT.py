# Classify tweets using LLAMA2-7B 
# (S.M.Iacus 2024)

import pandas as pd
from llama_cpp import Llama

finetuned_model = "llama2-7B-swb-FT-Q4_K_M.gguf"

n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
llm = Llama(
      model_path=finetuned_model,
      n_gpu_layers=-1, # Uncomment to use GPU acceleration
      seed=123, # Uncomment to set a specific seed
      n_ctx=1024, # Uncomment to increase the context window
      f16_kv=True, 
      n_batch=n_batch,
)



instruction = """Please classify the following text based on the well-being dimensions listed below. Use only the scale: 'low', 'medium', and 'high'. Return a JSON dictionary that contains only the well-being dimensions that apply. Do not explain your reasoning. 

Well-being dimensions:
[Happiness, Resilience, Self-esteem, Life satisfaction, Fear of future, Vitality, Having energy, Positive functioning, Expressing job satisfaction, Expressing optimism, Peace with thoughts and feelings, Purpose in life, Depression, Anxiety, Suffering, Feeling pain,
Expressing altruism, Loneliness, Quality of relationships, Belonging to society, Expressing gratitude, Expressing trust, Feeling trusted, Balance in the various aspects of own life, Mastery (ability or capability), Perceiving discrimination, Feeling loved by God, Belief in God, Religious criticism, Spiritual punishment, Feeling religious comfort,
Financial/material worry, Life after death belief, Volunteering, Charitable giving/helping, Seeking for forgiveness, Feeling having a political voice, Expressing government approval, Having hope, Promoting good, Expressing delayed gratification,
PTSD (Post-traumatic stress disorder), Describing smoking related health issues, Describing drinking related health issues, Describing health limitations, Expressing empathy].
"""

tweets = pd.read_csv("sample.csv")
tweets = tweets.rename({'message_id': 'id', 'tweet_text' : 'tweet'}, axis=1)
data = tweets[['id','tweet']]

def toDF(s, df):
    s = s.lower()
    s = s.replace("{", "").replace("}", "").replace("\n", "").split(",")   
    for i in s:
        try:
            v = [i.split(":")[0].strip('\'').replace("\"", "").replace("\'", ""), i.split(":")[1].strip('\'').replace("\"", "").replace(" ", "").replace("\'", "")]
        except:
            next
        else:
            df.loc[len(df)] = v #[i.split(":")[0].strip('\'').replace("\"", ""), i.split(":")[1].strip('\'').replace("\"", "").replace(" ", "")]

def analyze(thistext, id):
    prompt = f"""<s>[INST] {instruction}<s>Text: "{thistext}"[/INST]"""
    output = llm(prompt, max_tokens=512,echo=True,)
    answer = output["choices"][0]['text']
    answer = answer[answer.find('{'):]
    answer = answer[:answer.find('}')+1]
    locdf = pd.DataFrame(columns=['dimension', 'value'])       
    toDF(answer, locdf)
    locdf['id'] = id
    return(locdf)


mycsv = 're-classified-test.csv'

first = analyze(data.tweet[0], data.id[0])

first.to_csv(mycsv, mode='a', header=True, index=False)

for i in range(1,len(data)):
    id = data.loc[i, "id"]
    tweet = data.loc[i, "tweet"]
    print(f"\n\nN = {i}, id = {id}\ntweet = {tweet}\n")
    try:
        tmp = analyze(tweet, id)
    except:
        next
    else:
        print(tmp)
        tmp.to_csv(mycsv, mode='a', header=False, index=False)
        

exit(0)


