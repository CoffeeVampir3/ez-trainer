import gradio as gr

from tabs.train_tab import make_train_tab
from tabs.merge_lora_tab import make_merge_lora_tab
from mechanisms.train import initiate_training
from mechanisms.merge_lora import initiate_model_lora_merge

with gr.Blocks() as interface:
    with gr.Tab("Finetune Lora"):
        make_train_tab(launch_fn=initiate_training)
    with gr.Tab("Merge Model and Lora"):
        make_merge_lora_tab(launch_fn=initiate_model_lora_merge)
interface.launch()