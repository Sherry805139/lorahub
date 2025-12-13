import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

FULL_MODEL_PATH = "/home/hmpiao/hmpiao/Qwen2-VL-2B-Instruct"
LORA_ADAPTER_PATH = "/home/hmpiao/hmpiao/xuerong/FedMABench/output/category_lora_entertainment"
MERGED_WITH_LORA_PATH = "./output/lorahub_global_lora_merged"

def merge_lora_with_base_model():
    base_model_path = '${FULL_MODEL_PATH}'
    lora_adapter_path = '${LORA_ADAPTER_PATH}'
    merged_path = '${MERGED_WITH_LORA_PATH}'
    
    print(f'Loading base model from: {base_model_path}')
    print(f'Loading LoRA adapter from: {lora_adapter_path}')
    
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map='cpu'
        )
        
        print('Loading LoRA adapter...')
        model_with_lora = PeftModel.from_pretrained(base_model, lora_adapter_path, is_trainable=False)
        
        print('Merging LoRA adapter with base model...')
        merged_model = model_with_lora.merge_and_unload()
        
        print(f'Saving merged model to: {merged_path}')
        merged_model.save_pretrained(merged_path)
        
        print('Saving tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        tokenizer.save_pretrained(merged_path)
        
        print('LoRA adapter successfully merged with base model.')
        return True
        
    except Exception as e:
        print(f'Error merging LoRA adapter: {str(e)}')
        import traceback
        traceback.print_exc()
        return False