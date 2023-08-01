from tabs.tab_utils import get_available_from_dir, get_available_from_leafs
import gradio as gr
import os

def make_train_tab(launch_fn):
    with gr.Blocks() as interface:
        available_models = get_available_from_dir("models")
        available_datasets = get_available_from_leafs("datasets")
        available_loras = get_available_from_dir("loras")
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
            dataset_configs.append(gr.Slider(value = 0.10, minimum=-0.01, maximum=0.50, label="Validation Data Split"))
            with gr.Column():
                dataset_configs.append(gr.Textbox(value="text", lines=1, label='Dataset Sample ID'))
        with gr.Row():
            batch_size = gr.Slider(value=4, minimum=1, maximum=8192, label="Batch Size")
            training_configs = {
                "gradient_accumulation_steps" : gr.Number(value=2, label="Gradient Accumulation"),
                "num_train_epochs": gr.Number(value=2, label='Epochs'),
                "learning_rate": gr.Number(value=1e-4, label='Learning Rate'),
                "weight_decay": gr.Slider(value=0.10, minimum=0.00, maximum=1.0, label='Weight Decay'),
            }
        with gr.Row():
            training_configs["warmup_steps"] = gr.Number(value=7, label='Warmup Steps')
            training_configs["optim"] = gr.Dropdown(value='adamw_torch_fused', label='Optimizer', choices=['adamw_hf', 'adamw_torch', 'adamw_torch_fused', 'adamw_torch_xla', 'adamw_apex_fused', 'adafactor', 'adamw_bnb_8bit', 'adamw_anyprecision', 'sgd', 'adagrad'])
            training_configs["lr_scheduler_type"] = gr.Dropdown(value='constant_with_warmup', label='Learning Rate Scheduler Type', choices = ['linear', 'constant', 'constant_with_warmup', 'cosine', 'cosine_with_restarts', 'polynomial', 'inverse_sqrt'])
        with gr.Row():
            training_configs["output_dir"] = gr.Textbox(value="out", lines=1, label='Output Directory')
            training_configs["eval_steps"] = gr.Number(value=25, label='Evaluation Steps')
            training_configs["logging_steps"] = gr.Number(value=5, label='Logging Steps')
        
        submit = gr.Button("Submit")
        submit.click(fn=launch_fn, inputs=[model_path, lora_path, *lora_configs.values(), *dataset_configs, batch_size, *training_configs.values()], outputs=[gr.File()])
    return interface