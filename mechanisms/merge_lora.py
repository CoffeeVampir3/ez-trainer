from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from mechanisms.mech_utils import get_path_from_leaf
import sys, os

def merge(base_model, lora_model, scaling, merge_weight=1.0):
    weights_list = []

    # Loop over all parameters
    for name, param in lora_model.named_parameters():
        # If the parameter name ends with '.weight', it's an original weight
        if name.endswith('.weight'):
            # Make sure it's not a lora_A or lora_B weight
            if not any(substring in name for substring in ['lora_A', 'lora_B']):
                # Construct the names of the corresponding lora_A and lora_B weights
                layers = name.split('.')
                try:
                    layer = lora_model
                    for item in layers[:-1]:  # We go until the penultimate item (excluding the 'weight' part)
                        if 'lora' in item:  # Split further if lora_A or lora_B
                            item, lora_item = item.split('_')
                            layer = getattr(layer, item)
                            layer = getattr(layer, lora_item)
                        else:
                            layer = getattr(layer, item)
                        
                    # Try to get lora_A and lora_B weights
                    lora_A = getattr(layer, 'lora_A').default.weight
                    lora_B = getattr(layer, 'lora_B').default.weight

                    # Add a tuple to the list with the parameter name as the first item
                    weights_list.append((name, param.data, lora_A, lora_B))

                except AttributeError:
                    pass
                    #print(f"Unable to find lora_A or lora_B weights for {name}")

    for (name,weight,a,b) in weights_list:
        ab = b @ a
        weight += ab * scaling * merge_weight
        print(f"Did thing for layer named {name}")
    
    #clean lora loading trash
    for name, module in base_model.named_modules():
        if 'lora_A' in dir(module):
            delattr(module, 'lora_A')
        if 'lora_B' in dir(module):
            delattr(module, 'lora_B')

def get_lora_scaling(lora_model):
    r = lora_model.peft_config["default"].r
    alpha = lora_model.peft_config["default"].lora_alpha

    scaling = alpha/r
    return scaling

def load_model(model_path, lora_path):
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map = "auto",
    )

    print(f"Loading PEFT: {lora_path}")
    lora_model = PeftModel.from_pretrained(base_model, lora_path)
    
    return base_model, lora_model

def initiate_model_lora_merge(model_path, lora_path, output_dir, merge_weight):
    model_path = get_path_from_leaf("models", model_path)
    lora_path = get_path_from_leaf("loras", lora_path)
    
    print(model_path)
    print(lora_path)

    base_model, lora_model = load_model(model_path, lora_path)
    scaling = get_lora_scaling(lora_model)
    
    print(f"Lora Scaling: {scaling}")
    
    merge(base_model, lora_model, scaling, merge_weight=merge_weight)
    
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    final_model = base_model.save_pretrained(output_dir, use_safetensors=True)
    tokenizer.save_pretrained(output_dir)

    
    print("Done merging.")
    return final_model