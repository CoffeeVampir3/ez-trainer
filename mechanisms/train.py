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

def prep_trainer_init(
    model_init,
    tokenizer,
    train_dataset,
    training_configs
    ):
    args = TrainingArguments(
        **training_configs,
    )
    
    trainer = Trainer(
        model_init=model_init,
        train_dataset=train_dataset,
        args=args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    return trainer

def make_model(model_path, lora_config, gradient_checkpointing):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", load_in_4bit=True)
    model.gradient_checkpointing_enable()
    prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    lora_model = prep_model_for_lora(model, lora_config)
    if gradient_checkpointing:
        lora_model.enable_input_require_grads()
    lora_model.config.use_cache = False
    LOADED_MODEL = lora_model
    return LOADED_MODEL
    
def train_on(model_path, lora_path, lora_config, dataset_config, training_config):
    model_path = os.path.abspath(model_path)
    print(model_path)
    [print(f"{x}: {y}") for x,y in lora_config.items()]
    [print(f"{x}: {y}") for x,y in dataset_config.items()]
    [print(f"{x}: {y}") for x,y in training_config.items()]
    
    gradient_checkpointing = training_config["gradient_checkpointing"]
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    from functools import partial
    
    model_init = partial(make_model, model_path=model_path, lora_config=lora_config, gradient_checkpointing=gradient_checkpointing)
    lora_model = model_init()
    
    dataset = prep_dataset(tokenizer, **dataset_config)
    trainer = prep_trainer(lora_model, tokenizer, dataset["train"], training_config)
    
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
        "weight_decay":0.1,
        "do_train":True,
        "report_to":"none",
        
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