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
import wandb
from mechanisms.mech_utils import get_path_from_leaf
import bitsandbytes as bnb

def prep_dataset(tokenizer, dataset_path, dataset_type, dataset_validation_split, dataset_sample_id):
    dataset = load_dataset(dataset_type, data_files=dataset_path)
    dataset = dataset["train"].train_test_split(test_size=dataset_validation_split)
    data = dataset.map(lambda samples: tokenizer(samples[dataset_sample_id]), batched=True)
    
    train_data = data['train']
    test_data = data['test']
    return train_data, test_data

def prep_model_for_lora(model, lora_configs):
    config = LoraConfig(**lora_configs)
    lora_model = get_peft_model(model, config)
    return lora_model

def prep_trainer(
    model,
    tokenizer,
    train_dataset,
    test_dataset,
    training_configs
    ):
    args = TrainingArguments(
        **training_configs
    )
    
    trainer = Trainer(
        model = model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    return trainer

#from qlora and axolotl
def find_all_linear_names(bits, model):
    linear_class = torch.nn.Linear
    if bits == 4:
        linear_class = bnb.nn.Linear4bit
    if bits == 8:
        linear_class = bnb.nn.Linear8bitLt
        
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_class):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")

    return list(lora_module_names)

def make_model(model_path, lora_config, gradient_checkpointing):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", load_in_4bit=True)
    model.gradient_checkpointing_enable() 
    
    prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    
    target_modules = lora_config['target_modules']
    linear_layer_names = find_all_linear_names(4, model)
    target_modules_and_linears = list(set(target_modules + linear_layer_names))
    lora_config['target_modules'] = target_modules_and_linears
    
    [print(f"{x}: {y}") for x,y in lora_config.items()]
    
    lora_model = prep_model_for_lora(model, lora_config)
    if gradient_checkpointing:
        lora_model.enable_input_require_grads()
    lora_model.config.use_cache = False
    LOADED_MODEL = lora_model
    return LOADED_MODEL
    
def train_on(model_path, lora_path, lora_config, dataset_config, training_config):
    model_path = os.path.abspath(model_path)
    print(model_path)
    #[print(f"{x}: {y}") for x,y in lora_config.items()]
    [print(f"{x}: {y}") for x,y in dataset_config.items()]
    [print(f"{x}: {y}") for x,y in training_config.items()]
    
    gradient_checkpointing = training_config["gradient_checkpointing"]
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding=False, legacy=False, use_fast=False) ##Fast version is potentially bugged right now.
    tokenizer.pad_token = tokenizer.eos_token
    
    from functools import partial
    
    model_init = partial(make_model, model_path=model_path, lora_config=lora_config, gradient_checkpointing=gradient_checkpointing)
    lora_model = model_init()
    
    train, test = prep_dataset(tokenizer, **dataset_config)
    trainer = prep_trainer(lora_model, tokenizer, train, test, training_config)
    
    save_to = training_config["output_dir"]
    os.makedirs(save_to, exist_ok=True)
    
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

def initiate_training(
    model_path, lora_path, lora_r, lora_alpha, lora_dropout, 
    dataset_path, dataset_validation_split, dataset_sample_id, 
    batch_size, gradient_accumulation_steps, num_train_epochs, 
    learning_rate, weight_decay, warmup_steps, optim, lr_scheduler_type, output_dir, eval_steps, logging_steps):

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
        "dataset_validation_split": float(dataset_validation_split),
    }

    training_configs = {
        "save_strategy": IntervalStrategy.STEPS,
        "save_steps": 30,
        "save_total_limit": 5,
        "per_device_train_batch_size": int(batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "warmup_steps": int(warmup_steps),
        "num_train_epochs": int(num_train_epochs),
        "learning_rate": float(learning_rate),
        "bf16": True,
        "tf32": True,
        "optim": optim,
        "logging_steps": int(logging_steps),
        "evaluation_strategy": "steps",
        "eval_steps": int(eval_steps),
        "per_device_eval_batch_size": int(batch_size),
        "lr_scheduler_type": lr_scheduler_type,
        "output_dir": output_dir,
        "gradient_checkpointing": True,
        "weight_decay": float(weight_decay),
        "do_train":True,
        "report_to":"tensorboard",
        
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
        #"ddp_find_unused_parameters": None,
        
    }

    model_path = get_path_from_leaf("models", model_path)
    if lora_path:
        lora_path = get_path_from_leaf("loras", lora_path)

    return train_on(model_path, lora_path, lora_configs, dataset_configs, training_configs)