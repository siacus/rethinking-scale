# Transforms the csv file into a dataset for training LLAMA-2 models
# (S.M.Iacus 2024)

import json
import pandas as pd
import datasets

instruction = """Please classify the following text based on the well-being dimensions listed below. Use only the scale: 'low', 'medium', and 'high'. Return a JSON dictionary that contains only the well-being dimensions that apply. Do not explain your reasoning. 

Well-being dimensions:
[Happiness, Resilience, Self-esteem, Life satisfaction, Fear of future, Vitality, Having energy, Positive functioning, Expressing job satisfaction, Expressing optimism, Peace with thoughts and feelings, Purpose in life, Depression, Anxiety, Suffering, Feeling pain,
Expressing altruism, Loneliness, Quality of relationships, Belonging to society, Expressing gratitude, Expressing trust, Feeling trusted, Balance in the various aspects of own life, Mastery (ability or capability), Perceiving discrimination, Feeling loved by God, Belief in God, Religious criticism, Spiritual punishment, Feeling religious comfort,
Financial/material worry, Life after death belief, Volunteering, Charitable giving/helping, Seeking for forgiveness, Feeling having a political voice, Expressing government approval, Having hope, Promoting good, Expressing delayed gratification,
PTSD (Post-traumatic stress disorder), Describing smoking related health issues, Describing drinking related health issues, Describing health limitations, Expressing empathy].
"""

answers = pd.read_csv("answers.csv")
tweets = pd.read_csv("sample.csv")

tweets = tweets.rename({'message_id': 'id', 'tweet_text' : 'tweet'}, axis=1)

result = pd.merge(tweets, answers, on='id', how='inner')[['id','tweet','answer']]

def create_dataset(question, answer):
    return {
        "text": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }

import pandas as pd
from datasets import Dataset

def create_dataset(question, answer):
    text = f"""<s>[INST] {instruction} Text: "{question}"[/INST] {answer} </s>"""
    return text

# Apply the transformation
transformed_df = result.apply(lambda row: create_dataset(row['tweet'], row['answer']), axis=1)

# Convert the resulting Series to a DataFrame
transformed_df = transformed_df.to_frame(name='text')

# Convert the pandas DataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(transformed_df)

# Now 'dataset' is a Hugging Face Dataset
from huggingface_hub import HfApi, HfFolder

# Your Hugging Face token
hf_token = "hf_YOUR_HF_TOKEN"

# Save the token (this will save the token to the HfFolder's location, typically ~/.huggingface)
HfFolder.save_token(hf_token)

# Optionally, you can also set the token directly for the current script execution
HfApi().set_access_token(hf_token)


dataset.push_to_hub("siacus/tweets", split='train')


