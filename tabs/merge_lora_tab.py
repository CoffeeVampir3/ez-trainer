from tabs.tab_utils import get_available_from_dir, get_available_from_leafs, get_available_devices
import gradio as gr
import os

def make_merge_lora_tab(launch_fn):
    with gr.Blocks() as interface:
        available_models = get_available_from_dir("models")
        available_datasets = get_available_from_leafs("datasets")
        available_loras = get_available_from_dir("loras")
        available_devices = get_available_devices()
        default_device = available_devices[1] if len(available_datasets) > 0 else available_devices[0]
        
        with gr.Row():
            model_path = gr.Dropdown(choices = available_models, label="Base model")
            lora_path = gr.Dropdown(choices = available_loras, label="Resume from lora")
            target_devices = gr.Radio(value = default_device, choices = available_devices, label="Target Merging Devices", type="index")
        with gr.Row():
            output_dir = gr.Textbox(value="out", lines=1, label='Output Directory')
            merge_weight = gr.Slider(value=1.0, minimum=0.01, maximum=10.0, label="Merge Weight")
            
        submit = gr.Button("Submit")
        submit.click(fn=launch_fn, inputs=[model_path, lora_path, target_devices, output_dir, merge_weight], outputs=[gr.File()])
    return interface