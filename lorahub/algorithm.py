import os
import json
import copy
from functools import partial
from typing import List, Optional, Union, Any

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator, AutoTokenizer
from tqdm import tqdm
import pandas as pd
import numpy
import random
import nevergrad as ng

# 使用 swift 中封装过的 peft，自动打补丁以支持多模态模型（如 Qwen2-VL）
from swift.tuners import PeftModel, get_peft_model_state_dict
from peft.utils.save_and_load import set_peft_model_state_dict
from swift.llm.utils import get_model_tokenizer


def load_base_model_and_lora_modules(
    lora_module_list: List[str],
    model_name_or_path: Optional[str] = None,
    model_type: str = "qwen2-vl-2b-instruct",
):
    """加载 Qwen2-VL base model 和一组 LoRA 模块（来自本地 peft/Swift 训练的目录）

    Args:
        lora_module_list (List[str]): 一组 LoRA 目录（如 global_lora_30 或 category_lora_office 等）
        model_name_or_path (Optional[str]): 底模路径，默认从 LoRA 的 adapter_config.json 里读取
        model_type (str): swift 的 model_type 标识，默认 'qwen2-vl-2b-instruct'
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if len(lora_module_list) == 0:
        raise ValueError("lora_module_list 不能为空")

    # 以第一个 LoRA 目录为基准，读取 base_model_name_or_path
    default_peft_model_id = lora_module_list[0]

    # 直接解析本地 adapter_config.json，避免调用被 swift wrap 过的 PeftConfig.from_pretrained
    if model_name_or_path is None:
        adapter_cfg_path = os.path.join(default_peft_model_id, "adapter_config.json")
        if not os.path.exists(adapter_cfg_path):
            raise FileNotFoundError(f"adapter_config.json not found in {default_peft_model_id}")
        with open(adapter_cfg_path, "r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)
        model_name_or_path = adapter_cfg.get("base_model_name_or_path", None)
        if model_name_or_path is None:
            raise ValueError(
                f"'base_model_name_or_path' not found in {adapter_cfg_path}, "
                "please specify model_name_or_path explicitly."
            )

    # 使用 swift 的 get_model_tokenizer 正确加载 Qwen2-VL 模型（含 is_multimodal 标记等）
    # 这里强制将模型全部放到单块 GPU 上（cuda:0），避免 accelerate 的 device_map=auto
    # 把权重切到多块卡上导致输入张量 device 不一致的问题。
    single_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    base_model, tokenizer = get_model_tokenizer(
        model_type,
        torch_dtype=None,
        model_kwargs={"device_map": single_device},
        load_model=True,
        model_id_or_path=model_name_or_path,
        revision=None,
        quant_method=None,
    )

    # 默认 LoRA：构造一个带 LoRA 结构的 PeftModel（使用 swift 打过补丁的 peft）
    try:
        peft_model = PeftModel.from_pretrained(base_model, default_peft_model_id)
    except Exception as e:
        raise Exception(f"{default_peft_model_id} is unable to load into the model {model_name_or_path}: {e}")

    peft_model = peft_model.to(device)
    peft_model.eval()

    print("> Begin to load lora modules")
    cache = {}
    first_dict = None

    for peft_model_id in tqdm(lora_module_list):
        print("> Loading {} ...".format(peft_model_id))
        # 使用 PeftModel.from_pretrained 读取每个 LoRA 对应的 state_dict
        cur_peft_model = PeftModel.from_pretrained(base_model, peft_model_id)
        state_dict = copy.deepcopy(get_peft_model_state_dict(cur_peft_model))
        cache[peft_model_id] = state_dict

        if first_dict is None:
            first_dict = state_dict
        # 检查各个 LoRA 的结构是否一致（形状必须完全相同）
        try:
            for key in first_dict.keys():
                assert first_dict[key].shape == cache[peft_model_id][key].shape
        except Exception:
            raise Exception(
                f"LoRA Modules {peft_model_id} cannot be merged since it has a different arch (e.g., rank)."
            )

    # 将默认 LoRA 的权重加载到 peft_model 中，作为初始状态
    set_peft_model_state_dict(peft_model, cache[default_peft_model_id])

    return peft_model, tokenizer, cache

def preprocess_function(examples, tokenizer):
    """
    适配 Qwen2-VL 等 causal LM 的预处理：
    - 将 input 和 output 拼接成一个完整序列： full_text = input + "\\n\\n" + output
    - labels 与 input_ids 等长，但将 prompt 部分（对应 input）mask 掉（设为 -100）
    这样可以避免 seq2seq 风格下 input/label 长度不一致导致的 cross_entropy 维度错误。
    """
    inputs = examples["input"]
    targets = examples["output"]

    all_input_ids = []
    all_labels = []

    for inp, tgt in zip(inputs, targets):
        inp = inp if inp is not None else ""
        tgt = tgt if tgt is not None else ""
        full_text = inp + "\n\n" + tgt

        # 整体序列
        full_enc = tokenizer(
            full_text,
            max_length=2048,
            truncation=True,
            add_special_tokens=True,
        )
        input_ids = full_enc["input_ids"]

        # 只编码 prompt，用于确定需要 mask 的长度
        prompt_enc = tokenizer(
            inp,
            max_length=2048,
            truncation=True,
            add_special_tokens=True,
        )
        prompt_len = len(prompt_enc["input_ids"])

        labels = input_ids.copy()
        # 将 prompt 部分的 label 设为 -100，不参与损失
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        all_input_ids.append(input_ids)
        all_labels.append(labels)

    # 统一 padding，生成 batch
    batch = tokenizer.pad(
        {"input_ids": all_input_ids, "labels": all_labels},
        padding=True,
        max_length=2048,
        return_tensors="pt",
    )
    return batch


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
                      model_or_name_path: Union[Any, str],
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