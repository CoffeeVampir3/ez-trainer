import os, sys
import gradio as gr
from transformers import IntervalStrategy
import math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers.trainer_utils import FSDPOption
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from accelerate import infer_auto_device_map
import pathlib

def find_leaf_directories(parent_directory):
    leaf_directories = []
    for item in os.scandir(parent_directory):
        # If it's a directory, check if it has any visible subdirectories
        if item.is_dir():
            if all(name.startswith('.') or not os.path.isdir(os.path.join(item.path, name)) for name in os.listdir(item.path)):
                leaf_directories.append(item.path)
    return leaf_directories

def find_leaf_files(parent_directory):
    files = []
    for item in os.scandir(parent_directory):
        # If it's a file, add it to the list
        if item.is_file():
            files.append(item.path)
    return files
    
def get_path_from_leaf(root, leaf):
    return os.path.join(root, leaf)

def prep_dataset(tokenizer, dataset_path, dataset_type, dataset_sample_id):
    dataset = load_dataset(dataset_type, data_files=dataset_path)
    td = dataset
    data = td.map(lambda samples: tokenizer(samples[dataset_sample_id]), batched=True)
    
    return data

def prep_model_for_lora(model, lora_configs):
    config = LoraConfig(**lora_configs)
    lora_model = get_peft_model(model, config)
    return lora_model

def prep_trainer(
    model,
    tokenizer,
    train_dataset,
    training_configs
    ):
    args = TrainingArguments(
        **training_configs
    )
    
    trainer = Trainer(
        model = model,
        train_dataset=train_dataset,
        args=args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    return trainer
    
def train_on(model_path, lora_path, lora_config, dataset_config, training_config):
    print(model_path)
    [print(f"{x}: {y}") for x,y in lora_config.items()]
    [print(f"{x}: {y}") for x,y in dataset_config.items()]
    [print(f"{x}: {y}") for x,y in training_config.items()]
    
    gradient_checkpointing = training_config["gradient_checkpointing"]
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", load_in_4bit=True)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    lora_model = prep_model_for_lora(model, lora_config)
    
    dataset = prep_dataset(tokenizer, **dataset_config)
    trainer = prep_trainer(lora_model, tokenizer, dataset["train"], training_config)
    
    if gradient_checkpointing:
        lora_model.enable_input_require_grads()
    
    save_to = training_config["output_dir"]
    os.makedirs(save_to, exist_ok=True)
    
    lora_model.config.use_cache = False
    #lora_model = torch.compile(lora_model)
    
    if lora_path:
        print(f"Resuming from {lora_path}")
        trainer.train(resume_from_checkpoint=lora_path)
    else:
        print("Starting new training.")
        trainer.train()

    trainer.save_state()
    final_model = lora_model.save_pretrained(save_to)
    return final_model

def rebuild_dictionaries(
    model_path, lora_path, lora_r, lora_alpha, lora_dropout, 
    dataset_path, dataset_sample_id, 
    batch_size, per_device_train_batch_size, warmup_steps, num_train_epochs, 
    learning_rate, optim, logging_steps, lr_scheduler_type, output_dir):

    lora_configs = {
        'r': int(lora_r),
        'lora_alpha': int(lora_alpha),
        'target_modules': ["q_proj", "v_proj"],
        'lora_dropout': float(lora_dropout),
        "bias":"none", 
        "task_type":TaskType.CAUSAL_LM,
    }

    dataset_configs = {
        "dataset_path": get_path_from_leaf("datasets", dataset_path),
        "dataset_type":"json",
        "dataset_sample_id": dataset_sample_id,
    }

    training_configs = {
        "save_strategy": IntervalStrategy.STEPS,
        "save_steps": 30,
        "save_total_limit": 5,
        "per_device_train_batch_size": int(per_device_train_batch_size),
        "gradient_accumulation_steps": int(batch_size//per_device_train_batch_size),
        "warmup_steps": int(warmup_steps),
        "num_train_epochs": int(num_train_epochs),
        "learning_rate": float(learning_rate),
        "bf16": True,
        "tf32": True,
        "optim": optim,
        "logging_steps": int(logging_steps),
        "evaluation_strategy": "no",
        "lr_scheduler_type": lr_scheduler_type,
        "ddp_find_unused_parameters": None,
        "output_dir": output_dir,
        "gradient_checkpointing": True,
        
        ### Future stuff for distributed training.
        #"fsdp":"full_shard auto_wrap",
        #"fsdp_transformer_layer_cls_to_wrap": 'LlamaDecoderLayer',
        #"local_rank":local_rank,
        # "fsdp":FSDPOption.FULL_SHARD,
        # "fsdp_config":{
        #     "fsdp_min_num_params": 0,
        #     "fsdp_backward_prefetch": "backward_pre",
        #     "fsdp_forward_prefetch": True,
        #     "limit_all_gathers": False,
        # },
        #"deepspeed":"deepspeed.json"
        
    }

    model_path = get_path_from_leaf("models", model_path)
    if lora_path:
        lora_path = get_path_from_leaf("loras", lora_path)

    return train_on(model_path, lora_path, lora_configs, dataset_configs, training_configs)

with gr.Blocks() as interface:
    available_models = [os.path.basename(x) for x in find_leaf_directories("models")]
    available_datasets = [os.path.basename(x) for x in find_leaf_files("datasets")]
    available_loras = [os.path.basename(x) for x in find_leaf_directories("loras")]
    model_path = gr.Dropdown(choices = available_models, label="Base model")
    lora_path = gr.Dropdown(choices = available_loras, label="Resume from lora")
    
    with gr.Row():
        lora_configs = {
            'r': gr.Slider(value=8, minimum=1, maximum=1024, label="Lora Rank"),
            'lora_alpha': gr.Slider(value=8, minimum=1, maximum=2048, label="Lora Alpha"),
            'lora_dropout': gr.Slider(value = 0.00, minimum = 0.0, maximum=1.0, label="Lora Dropout"),
        }
        
    with gr.Row():
        dataset_configs = []
        dataset_configs.append(gr.Dropdown(label='Dataset File', choices=available_datasets))
        with gr.Column():
            dataset_configs.append(gr.Textbox(value="text", lines=1, label='Dataset Sample ID'))
    with gr.Row():
        batch_size = gr.Slider(value=4, minimum=1, maximum=8192, label="Batch Size")
        training_configs = {
            "per_device_train_batch_size": gr.Number(value=2, label='Sub Batch Size'),
            "warmup_steps": gr.Number(value=7, label='Warmup Steps'),
            "num_train_epochs": gr.Number(value=2, label='Epochs'),
            "learning_rate": gr.Number(value=1e-4, label='Learning Rate'),
            "optim": gr.Dropdown(value='adamw_torch_fused', lines=1, label='Optimizer', choices=['adamw_hf', 'adamw_torch', 'adamw_torch_fused', 'adamw_torch_xla', 'adamw_apex_fused', 'adafactor', 'adamw_bnb_8bit', 'adamw_anyprecision', 'sgd', 'adagrad']),
            "logging_steps": gr.Number(value=5, label='Logging Steps'),
            "lr_scheduler_type": gr.Dropdown(value='constant_with_warmup', label='Learning Rate Scheduler Type', choices = ['linear', 'constant', 'constant_with_warmup', 'cosine', 'cosine_with_restarts', 'polynomial', 'inverse_sqrt']),
            "output_dir": gr.Textbox(value="out", lines=1, label='Output Directory'),
        }
    
    submit = gr.Button("Submit")
    submit.click(fn=rebuild_dictionaries, inputs=[model_path, lora_path, *lora_configs.values(), *dataset_configs, batch_size, *training_configs.values()], outputs=[gr.File()])
    
interface.launch()