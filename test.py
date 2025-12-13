import os
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

FULL_MODEL_PATH = "/home/hmpiao/hmpiao/Qwen2-VL-2B-Instruct"
LORA_ADAPTER_PATH = "/home/hmpiao/hmpiao/xuerong/FedMABench/output/category_lora_entertainment"
MERGED_WITH_LORA_PATH = "./output/lorahub_global_lora_merged"

def merge_lora_with_base_model():
    # 使用上面的 Python 变量，而不是字符串模板
    base_model_path = FULL_MODEL_PATH
    lora_adapter_path = LORA_ADAPTER_PATH
    merged_path = MERGED_WITH_LORA_PATH
    
    print(f'Loading base model from: {base_model_path}')
    print(f'Loading LoRA adapter from: {lora_adapter_path}')
    
    try:
        # Qwen2-VL 是多模态模型，不能用 AutoModelForCausalLM，而是要用 Qwen2VLForConditionalGeneration
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map='cpu'
        )
        print(f'Base model class: {type(base_model)}')
        
        print('Loading LoRA adapter...')
        model_with_lora = PeftModel.from_pretrained(base_model, lora_adapter_path, is_trainable=False)
        print(f'Model with LoRA class: {type(model_with_lora)}')

        # 抽样记录一部分浮点参数，方便对比 merge 前后是否发生变化
        sampled_param_names = []
        base_snapshots = {}
        for name, param in base_model.state_dict().items():
            if torch.is_floating_point(param):
                sampled_param_names.append(name)
                base_snapshots[name] = param.detach().cpu().clone()
                if len(sampled_param_names) >= 10:
                    break
        print(f"Sampled {len(sampled_param_names)} float parameters for diff check.")
        
        print('Merging LoRA adapter with base model...')
        merged_model = model_with_lora.merge_and_unload()
        print(f'Merged model class: {type(merged_model)}')

        # 对比 merge 前后这些参数是否发生了变化
        merged_state = merged_model.state_dict()
        for name in sampled_param_names:
            before = base_snapshots[name]
            after = merged_state[name].detach().cpu()
            diff = (after - before).abs().max().item()
            print(f'Param "{name}": max abs diff = {diff:.6e}')
        
        print(f'Saving merged model to: {merged_path}')
        merged_model.save_pretrained(merged_path)
        
        print('Saving tokenizer...')
        # 对于 Qwen2-VL，多模态输入推荐使用 AutoProcessor
        tokenizer = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
        tokenizer.save_pretrained(merged_path)
        
        print('LoRA adapter successfully merged with base model.')
        return True
        
    except Exception as e:
        print(f'Error merging LoRA adapter: {str(e)}')
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    merge_lora_with_base_model()