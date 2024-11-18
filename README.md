# Rethinking Scale
Supplementary material for Rethinking Scale: The Efficacy of Fine-Tuned Open-Source LLMs in Large-Scale Reproducible Social Science Research


# System configuration

Setup [conda](https://docs.anaconda.com/miniconda/)

Then run these commands `conda create -n rethink python=3.10`

`conda activate rethink`

Then pip install the following libraries

`pip3 install accelerate peft bitsandbytes transformers trl`

`pip install huggingface-hub`

Any time you run one of the scripts in [dataverse](dataverse), [cap](cap) or [flourishing](flourishing) do not forget to activate the `rethink` envornment.

Also execute `huggingface-cli login` from the terminal to authenticate to the [Huggingface](https://huggingface.co) portal.

All scripts can be run through this approach"
`nohup python -u myscript.py > log-myscript.txt &!`

If you also want to use [wandb](https://wandb.ai/) to log the progress of the fine-tuning, you need to create an account first and pass this information to the script.
