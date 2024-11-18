# Rethinking Scale
Supplementary material for [Rethinking Scale: The Efficacy of Fine-Tuned Open-Source LLMs in Large-Scale Reproducible Social Science Research](https://arxiv.org/abs/2411.00890)


# System configuration

Setup [conda](https://docs.anaconda.com/miniconda/)

Then run these commands 

`conda create -n rethink python=3.10`

`conda activate rethink`

Then pip install the following libraries

`pip3 install accelerate peft bitsandbytes transformers trl`

`pip install huggingface-hub`

Any time you run one of the scripts in [dataverse](dataverse), [cap](cap) or [flourishing](flourishing) do not forget to activate the `rethink` envornment.

Also execute `huggingface-cli login` from the terminal to authenticate to the [Huggingface](https://huggingface.co) portal.

All scripts can be run through this approach:

`nohup python -u myscript.py > log-myscript.txt &!`

If you also want to use [wandb](https://wandb.ai/) to log the progress of the fine-tuning, you need to create an account first and pass this information to the script.

# Fine-tuning 
Due to memory limitations, we always fine-tuned the models and run inference from those using quantization.

Because the libraries used rely on [bitsandbites](https://github.com/bitsandbytes-foundation/bitsandbytes) that does not yet support (at the time of our experiments) quantization on MAC M1/M2 architectures, we execute fine-tuning on either A100 or H100 CUDA GPUs.


# Merging the weights
We use the standard approach. One example of working script is given [here](dataverse/merge-weights-small.py) for the dataverse case.

# Quantization
Most of the times we quantized the models directly on an M2 Ultra server after merging the weights and creating f16 versions of the weights using [llama.cpp](https://github.com/ggerganov/llama.cpp) on the CUDA GPUs.

The next sequence of commands assumes you have cloned [llama.cpp](https://github.com/ggerganov/llama.cpp), built it and have the commands available in the path on both the CUDA GPUs machines and the Metal M2 machines.

On the CUDA GPUs machines we first create the F16 GGUF file from the merged models

`path_to_llama.cpp/convert-hf-to-gguf.py llama-2-7b-small-dv \
    --outfile llama-2-7b-small-dv-f16.gguf \
    --outtype f16`

and on the M2 Ultra (or the CUDA GPUs) quantize to 4bit:

`path_to_llama.cpp/llama.cpp/llama-quantize llama-2-7b-small-dv-f16.gguf \
    llama-2-7b-small-dv-Q4_K_M.gguf  Q4_K_M` 

In our experience it turned out that it was important to get the all up-to-date version of llama.cpp before quantizing. 

It is also very important to quantize to 4bit on the architecture you are going to run the inference, i.e., move the F16 GGUF file to M2 Ultra machine and quantize it there to 4bit, or do everything on CUDA GPUs. Do note mix up architectures: i.e., quantize to 4bit on CUDA and use the 4bit-qauntized GGUF on a M2 Ultra. Results can be a disaster!!!

