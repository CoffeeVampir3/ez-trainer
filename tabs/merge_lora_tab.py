from tabs.tab_utils import get_available_from_dir, get_available_from_leafs
import gradio as gr
import os

def make_merge_lora_tab(launch_fn):
    with gr.Blocks() as interface:
        available_models = get_available_from_dir("models")
        available_datasets = get_available_from_leafs("datasets")
        available_loras = get_available_from_dir("loras")
        model_path = gr.Dropdown(choices = available_models, label="Base model")
        lora_path = gr.Dropdown(choices = available_loras, label="Merge Lora")
        output_dir = gr.Textbox(value="out", lines=1, label='Output Directory')
        merge_weight = gr.Slider(value=1.0, minimum=0.01, maximum=10.0, label="Merge Weight")
        submit = gr.Button("Submit")
        submit.click(fn=launch_fn, inputs=[model_path, lora_path, output_dir, merge_weight], outputs=[gr.File()])
    return interface