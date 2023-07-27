# ez-trainer
Train Llama Loras Easily

Extremely ugly interface for easy lora training for Llama models. Put your models in the models folder, put your datasets in the datasets folder. Wow!

Datasets are expected to be json or jsonl format right now. Models should be the unquantized huggingface format. More info soon, maybe.

# Check out the wiki for usage info

<https://github.com/CoffeeVampir3/ez-trainer/wiki/Train-the-Stuff-and-Stuff#crash-course>

## Install:

`pip install -r requirements.txt`

(This does not install torch, you should install that yourself. <https://pytorch.org/get-started/locally/>)

(Step by step instructions for exactly how to install in the wiki <https://github.com/CoffeeVampir3/ez-trainer/wiki/Train-the-Stuff-and-Stuff#crash-course>)

## Run:

`python train_module.py`
