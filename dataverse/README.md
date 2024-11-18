# Dataverse
The data used to train the model are on Huggingface under [siacus/dv_subject](https://huggingface.co/datasets/siacus/dv_subject)

The `small-dv` version of the fine-tuned model works on a training-set of 5,000 randomly sampled data.

The large version works on the whole 76.1K training records.

The test set is of size 32.6K rows.

The two versions of the fine-tuned models in GGUF format in both F16 and 4bit versions can be obtained from Hugginface: [llama-2-7b-small-dv](https://huggingface.co/siacus/llama-2-7b-small-dv) and [llama-2-7b-dv](https://huggingface.co/siacus/llama-2-7b-dv)

# Prompt used
The prompt here is composed of the Title and the Description of the dataset. The fine-tuned model is supposed to answer using a json list like in the following example:

`<s>[INST] Here are Title and Description of a dataset. Title :'A15_29234.jpg' Description:'Link to OCHRE database: <a href = 'http://pi.lib.uchicago.edu/1001/org/ochre/526ee9d3-6b44-44b9-b407-adebe96fcf82'>http://pi.lib.uchicago.edu/1001/org/ochre/526ee9d3-6b44-44b9-b407-adebe96fcf82'. And this are subject categories: ['Medicine, Health and Life Sciences','Arts and Humanities','Computer and Information Science','Social Sciences','Mathematical Sciences','Physics','Earth and Environmental Sciences','Chemistry','Engineering','Other','Law','Business and Management','Astronomy and Astrophysics','Agricultural Sciences']. Please return only a json list of at most 3 elements corresponding to the text labels:[/INST] ['Arts and Humanities']. </s>`
