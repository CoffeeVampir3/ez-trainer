# ez-trainer
Train Llama Loras Easily

Extremely ugly interface for easy lora training for Llama models. Put your models in the models folder, put your datasets in the datasets folder. Wow!

Datasets are expected to be json or jsonl format right now. Models should be the unquantized huggingface format. More info soon, maybe.

## Install:

`pip install -r requirements.txt`

(This does not install torch, you should install that yourself. <https://pytorch.org/get-started/locally/>)

## Run:

`python train_module.py`
