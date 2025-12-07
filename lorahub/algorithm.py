import os
from transformers import AutoTokenizer
from transformers import Qwen2VLForConditionalGeneration
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator
from tqdm import tqdm
import pandas as pd
import numpy
import random
import nevergrad as ng
from peft.utils.save_and_load import set_peft_model_state_dict, get_peft_model_state_dict
from peft import PeftModel, PeftConfig, get_peft_model
from functools import partial
from typing import List, Optional, Union
import copy

try:
    # safetensors 加载 LoRA 权重（如果可用）
    from safetensors.torch import load_file as safetensors_load_file
except ImportError:
    safetensors_load_file = None

def load_base_model_and_lora_modules(lora_module_list: List[str], model_name_or_path: Optional[str] = None):
    """load base model and lora modules from huggingface model hub

    Args:
        lora_module_list (List[str]): a list of lora module names available in huggingface model hub
        model_name_or_path (Optional[str]): base model name, default is None
    """
    # use gpu if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load basic model
    default_peft_model_id = lora_module_list[0]

    # 读取默认 LoRA 的 peft 配置
    default_peft_config: PeftConfig = PeftConfig.from_pretrained(default_peft_model_id)

    # find the base model
    if model_name_or_path is None:
        model_name_or_path = default_peft_config.base_model_name_or_path

    # 对于 Qwen2-VL-2B-Instruct 这类多模态因果模型，直接使用 Qwen2VLForConditionalGeneration 加载
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    # load tokenizer（同样加上 trust_remote_code 以兼容 Qwen2-VL）
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )

    # 修正 Qwen2-VL LoRA 的 target_modules，避免把整个 Qwen2VLModel 当作 target module
    # 原始配置中 target_modules 通常是 "^(model)(?!.*(lm_head|output|emb|wte|shared)).*"
    # 这里将其收窄到 language_model 子模块
    if hasattr(default_peft_config, "target_modules") and isinstance(default_peft_config.target_modules, str):
        if default_peft_config.target_modules.startswith("^(model)"):
            default_peft_config.target_modules = r"^model\.language_model(?!.*(lm_head|output|emb|wte|shared)).*"

    # 基于修正后的配置构建带 LoRA 结构的 PeftModel（此时还未加载具体权重）
    try:
        peft_model = get_peft_model(base_model, default_peft_config)
    except Exception as e:
        raise Exception(f"Failed to create PeftModel for {default_peft_model_id}: {e}")

    def _load_lora_state_dict(peft_model_id: str):
        """
        从本地目录加载 LoRA adapter 的 state_dict，优先使用 safetensors。
        """
        if os.path.isdir(peft_model_id):
            # 优先 adapter_model.safetensors
            if safetensors_load_file is not None:
                st_path = os.path.join(peft_model_id, "adapter_model.safetensors")
                if os.path.exists(st_path):
                    return safetensors_load_file(st_path, device="cpu")
            # 其次 adapter_model.bin
            bin_path = os.path.join(peft_model_id, "adapter_model.bin")
            if os.path.exists(bin_path):
                return torch.load(bin_path, map_location="cpu")

        # 回退：使用 PeftModel.from_pretrained（例如 HuggingFace Hub 模型）
        tmp_peft_model = PeftModel.from_pretrained(base_model, peft_model_id)
        return copy.deepcopy(get_peft_model_state_dict(tmp_peft_model))

    print("> Begin to load lora modules")
    cache = {}
    first_dict = None

    for peft_model_id in tqdm(lora_module_list):
        print("> Loading {} ...".format(peft_model_id))
        state_dict = _load_lora_state_dict(peft_model_id)
        cache[peft_model_id] = state_dict

        if first_dict is None:
            first_dict = state_dict
        # check whether the LoRA can be merged into one
        try:
            # detect whether the arch is the same
            for key in first_dict.keys():
                assert first_dict[key].shape == cache[peft_model_id][key].shape
        except Exception:
            raise Exception(
                f'LoRA Modules {peft_model_id} cannot be merged since it has a different arch (e.g., rank).'
            )

    # 将默认 LoRA 的权重加载到 peft_model 中，作为初始状态
    set_peft_model_state_dict(peft_model, cache[default_peft_model_id])

    peft_model = peft_model.to(device)
    peft_model.eval()

    return peft_model, tokenizer, cache

def preprocess_function(examples, tokenizer):
    """
    standard preprocess function for dataset
    """
    inputs = examples["input"]
    targets = examples["output"]
    model_inputs = tokenizer(
        inputs,
        max_length=2048,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = tokenizer(
        targets,
        max_length=2048,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs


def load_dataset(example_inputs, example_outputs, tokenizer):
    # add empty string if example_outputs is None
    if example_outputs is None:
        example_outputs = [""] * len(example_inputs)
    df = [
        {"input": example_inputs[i], "output": example_outputs[i]}
        for i in range(len(example_inputs))
    ]
    dataset = Dataset.from_pandas(pd.DataFrame(df))
    preprocess_func_with_tokenizer = partial(preprocess_function, tokenizer=tokenizer)
    processed_datasets = dataset.map(
        preprocess_func_with_tokenizer,
        batched=True,
        num_proc=1,
        desc="Running tokenizer on dataset",
    )
    return processed_datasets


def default_get_loss(example_dataset, model, batch_size):
    """
    Get the loss of the model on the example dataset. Usually the example dataset only contains a few examples.
    """
    data_batch_size = len(example_dataset) if batch_size is None else min(len(example_dataset), batch_size)
    # use gpu if available
    train_dataloader = DataLoader(
        example_dataset,
        collate_fn=default_data_collator,
        batch_size=data_batch_size,
        pin_memory=True,
    )
    train_loss = 0
    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for _, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            train_loss += loss.detach().float()
    loss = train_loss.float()
    # average loss over the number of examples
    return float(loss) / len(example_dataset["input"])

def default_l1_regularization(weights):
    """
    Get the L1 regularization term for the weights
    """
    sum_of_squares = sum([abs(x) for x in weights]) / len(weights)
    return 0.05 * sum_of_squares

def get_score(weights, model, cache, example_dataset, batch_size, get_loss, get_regular):
    # the composed lora state dict
    final_state_dict = {}
    # module list is the list
    lora_module_list = list(cache.keys())
    # all keys are the same
    keys = cache[lora_module_list[0]].keys()
    for i, peft_model_id in enumerate(lora_module_list):
        lora_state_dict = cache[peft_model_id]
        if i == 0:
            for key in keys:
                final_state_dict[key] = weights[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + weights[i] * lora_state_dict[key]
                )
    # reload the model with the new adapter config
    set_peft_model_state_dict(model, final_state_dict)
        
    # minimize the metric
    loss = get_loss(example_dataset, model, batch_size)
    # L1 regularization term
    metric_val = loss + get_regular(weights)
    
    return metric_val

def get_final_weights(weights, lora_module_list, cache):
    final_state_dict = {}
    keys = cache[lora_module_list[0]].keys()
    for i, peft_model_id in enumerate(lora_module_list):
        lora_state_dict = cache[peft_model_id]
        if i == 0:
            for key in keys:
                final_state_dict[key] = weights[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + weights[i] * lora_state_dict[key]
                )
    return final_state_dict
    
def lorahub_inference(example_inputs: List[str],
                      model_or_name_path: Union[Qwen2VLForConditionalGeneration, str],
                      tokenizer_or_tokenizer_path: Union[AutoTokenizer, str],
                      batch_size: int,
                      # if not provided, we do not report the accuracy
                      example_outputs: List[str]=None):
    
    def accuracy_score(outputs, ground_truths):
        correct = 0
        total = 0
        for output, truth in zip(outputs, ground_truths):
            if output.strip().lower().replace(".", "") == truth.strip().lower().replace(".", ""):
                correct += 1
            total += 1
        return correct / total * 100

    example_predictions = []
    # load model（支持直接传路径或已加载好的模型）
    if isinstance(model_or_name_path, str):
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_or_name_path,
            trust_remote_code=True,
        )
    else:
        model = model_or_name_path
    
    # load tokenizer
    if isinstance(tokenizer_or_tokenizer_path, str):
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_or_tokenizer_path,
            trust_remote_code=True,
        )
    else:
        tokenizer = tokenizer_or_tokenizer_path
            
    # process dataset
    dataset = load_dataset(example_inputs, example_outputs, tokenizer)
    # use gpu if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    for i in range(0, len(dataset["input"]), batch_size):
        inputs = tokenizer(
            dataset["input"][i : i + batch_size],
            max_length=2048,
            return_tensors="pt",
            padding=True,
        ).to(device)
        outputs = model.generate(
            input_ids=inputs["input_ids"], max_new_tokens=256
        )
        outputs = tokenizer.batch_decode(
            outputs.to("cpu"), skip_special_tokens=True
        )
        example_predictions.extend(outputs)
    
    if example_outputs is not None:
        task_perf = accuracy_score(example_predictions, example_outputs)
    else:
        task_perf = None
        
    return example_predictions, task_perf


def lorahub_learning(lora_module_list: List[str], 
                     example_inputs: List[str], 
                     example_outputs: List[str], 
                     max_inference_step: int,
                     model_name_or_path=None,
                     batch_size=None,
                     get_loss=default_get_loss, 
                     get_regular=default_l1_regularization,
                     seed=42):
    # set seed for reproducibility
    random.seed(seed)
    numpy.random.seed(seed)

    number_of_loras = len(lora_module_list)
    if number_of_loras == 0:
        print("> No LoRA modules are provided. Please provide at least one LoRA module.")
        return None, None

    # load model
    model, tokenizer, cache = load_base_model_and_lora_modules(lora_module_list, model_name_or_path)
    # process dataset
    dataset = load_dataset(example_inputs, example_outputs, tokenizer) 
    get_score_partial = partial(get_score, 
                                model=model, 
                                cache=cache,
                                example_dataset=dataset,
                                batch_size=batch_size,
                                get_loss=get_loss, 
                                get_regular=get_regular)
    # set up the limit of the weights
    instrum = ng.p.Array(
        init=[0] * number_of_loras,
        upper=[1.5] * number_of_loras,
        lower=[-1.5] * number_of_loras,
    )
    optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=max_inference_step)
    print("> Begin to perform gradient-free optimization ...")
    recommendation = optimizer.minimize(get_score_partial, verbosity=1)
    final_lora = get_final_weights(recommendation.value, lora_module_list, cache)
    # set the final weights
    set_peft_model_state_dict(model, final_lora)
    model = model.merge_and_unload()
    return recommendation.value, model, tokenizer