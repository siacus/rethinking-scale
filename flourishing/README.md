# Human Flourishing
The models and data used to train and test the different LLAMA models are contained in this directory.
All base omdels should be downloaded locally to run the scripts.

# Base models
For the base LLAMA2 models we rely on these GGUF quantized versions:

1. [LLAMA-2-7B](https://huggingface.co/TheBloke/Llama-2-7B-GGUF) : download model `llama-2-7b-chat.Q4_K_M.gguf`
2. [LLAMA-2-13B](https://huggingface.co/TheBloke/Llama-2-13B-GGUF) : download model `llama-2-13b-chat.Q4_K_M.gguf`
3. [LLAMA-2-70B](https://huggingface.co/TheBloke/Llama-2-70B-GGUF) : download model `llama-2-70b-chat.Q4_K_M.gguf`

For the base LLAMA3 models we rely on these GGUF quantized versions:

4. [LLAMA-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) : download model `Meta-Llama-3-8B-Instruct.Q4_K_M.gguf`
5. [LLAMA-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) : download model `Meta-Llama-3-70B-Instruct.Q5_K_M.gguf`

For the base LLAMA3.1 models we rely on these GGUF quantized versions:

6. [LLAMA-3.1-405B](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct) : download model `Meta-Llama-3.1-405B-Instruct`

For the base LLAMA3.2 models we rely on these GGUF quantized versions:

7. [LLAMA-3.2-1B](https://huggingface.co/jxtngx/Meta-Llama-3.2-1B-Instruct-Q4_K_M-GGUF) : download model `Llama-32-1B-Q4_K_M.gguf`
8. [LLAMA-3.2-3B](https://huggingface.co/jxtngx/Meta-Llama-3.2-3B-Instruct-Q4_K_M-GGUF) : download model `Llama-32-3B-Q4_K_M.gguf`


# Fine-tuning
These python scripts perform the training for the LLAMA-2 and LLAMA-3 models. We only produce one example per family of models.

1. [finetune-L2-13B.py](finetune-L2-13B.py) : fine-tuning LLAMA-2 models
2. [finetune-L32-3B.py](finetune-L32-3B.py) : fine-tuning LLAMA-3 models

# Prompt used
The prompt is splitted into four similar prompts and each tweets is analyzed four times, to keep the number of dimensions reasonable. The next is a template prompt used in LLAMA-3 models:

`template1 = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> {instruction1} <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the text: \n\n {{text}} \n Please, return a JSON dictionary = <|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
`

The human flourishing dimensions:

`dims1 ="""[Happiness, Resilience, Self-esteem, Life satisfaction, Fear of future, Vitality, Having energy, Positive functioning, Expressing job satisfaction, Expressing optimism, Peace with thoughts and feelings, Purpose in life, Depression, Anxiety, Suffering, Feeling pain]"""`

`dims2 ="""[Expressing altruism, Loneliness, Quality of relationships, Belonging to society, Expressing gratitude, Expressing trust, Feeling trusted, Balance in the various aspects of own life, Mastery (ability or capability), Perceiving discrimination, Feeling loved by God, Belief in God, Religious criticism, Spiritual punishment, Feeling religious comfort]"""`

`dims3 ="""[Financial or material worry, Life after death belief, Volunteering, Charitable giving/helping, Seeking for forgiveness, Feeling having a political voice, Expressing government approval, Having hope, Promoting good, Expressing delayed gratification]"""`

`dims4 ="""[PTSD (Post-traumatic stress disorder), Describing smoking related health issues, Describing drinking related health issues, Describing health limitations, Expressing empathy]"""`

Example of template for the LLAMA-2 models:



# Datasets
Datasets have been stored to Huggingface. We have two versions that include prompting:

1. [siacus/tweets](https://huggingface.co/datasets/siacus/tweets) : for traiing LLAMA-2 models

2. [siacus/train-llama3](https://huggingface.co/datasets/siacus/train-llama3) : for traiing LLAMA-3 models

And the original data used to generate the training sets are in [here](sample.csv) and the test set data are [here](answers.csv).

3. [generate-L2-trainingset.py](generate-L2-trainingset.py) : creates the dataset `siacus/tweets` on Huggingface with training set and test set splits for LLAMA-2 models

4. [generate-L3-trainingset.py](generate-L3-trainingset.py) : creates the dataset `siacus/train-llama3` on Huggingface with training set and test set splits for LLAMA-3 models



# Fine-tuned models 
The fine-tuned models can be obtained on request. Due to their large size (and lower accuracy) we do not redistribute them on Huggingface. More models will be added in the future if space on Huggingface will allow for this. 
Here we point the user to the best among the fine-tuned models. The model is stored on Huggingface as [LLAMA-7B parameter model](https://huggingface.co/siacus/llama2-7B-swb-FT-Q4_K_M.gguf).



# Inference
Run these python scripts to generate inference for the differet models. Each model produces a .csv file with the classification. Here we report two examples for the the two families of models.

1. [classify-L2-7B.py](classify-L2-7B.py) : generates [classified-L2-7B.csv](classified-L2-7B.csv) unsing the LLAMA-2-7B base model

2. [classify-L32-3B.py](classify-L32-3B.py) : generates [classified-L2-7B.csv](classified-L32-3B.csv) using the LLAMA-3.2-8B base model

3. [classify-L2-7B-FT.py](classify-L2-7B-FT.py) : generates [re-classified-test.csv](re-classified-test.csv) using the fine-tuned model


# Analysis
The file [models.csv](models.csv) contain the list of model names, their parameters and classification output used to generate the tables in the manuscript.



# Summary statistics
`change this with the actual script`
Summary statistics can be obtained executing this R script: [createStats.R](createStats.R)

