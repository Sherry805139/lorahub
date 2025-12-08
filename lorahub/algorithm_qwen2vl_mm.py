import os
from typing import List, Optional, Union, Any, Tuple

import numpy as np
import random
import torch
from PIL import Image
from tqdm import tqdm

import nevergrad as ng
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)

from qwen_vl_utils import process_vision_info

from .algorithm import (
    load_base_model_and_lora_modules,
    default_l1_regularization,
    get_final_weights,
)
from peft.utils.save_and_load import set_peft_model_state_dict


# 为了控制显存占用，限制每个样本参与建模的图片数量和文本长度
MAX_TEXT_LENGTH = 1024


def _set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_text_and_image_paths(text: str) -> Tuple[str, List[str]]:
    """
    从 input 文本中解析出纯文本部分和图片路径列表。

    约定：图片路径以
        "\\n\\n[IMAGE_PATHS]\\n" 作为分隔符，其后每行一个绝对/相对路径。
    """
    if "[IMAGE_PATHS]" not in text:
        return text.strip(), []

    prefix, paths_block = text.split("[IMAGE_PATHS]", 1)
    prefix = prefix.strip()
    paths_block = paths_block.strip()
    image_paths: List[str] = []
    for line in paths_block.splitlines():
        line = line.strip()
        if not line:
            continue
        image_paths.append(line)
    return prefix, image_paths


def _load_images(image_paths: List[str]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for p in image_paths:
        try:
            if os.path.exists(p):
                img = Image.open(p).convert("RGB")
                images.append(img)
        except Exception:
            # 避免因为单张图片坏掉而中断整个 batch
            continue
    return images


def _build_mm_messages_and_images(
    input_text: str,
    output_text: Optional[str] = None,
) -> Tuple[List[dict], List[Image.Image]]:
    """
    将 LoraHub 的 input / output 文本转成 Qwen2-VL 官方推荐的多模态对话格式：

    messages = [
        {"role": "user", "content": [{"type": "text", "text": ...},
                                     {"type": "image", "image": PIL.Image}, ...]},
        {"role": "assistant", "content": [{"type": "text", "text": output_text}]}
    ]
    """
    user_text, image_paths = _parse_text_and_image_paths(input_text or "")
    images = _load_images(image_paths)

    user_content = [{"type": "text", "text": user_text}]
    for img in images:
        user_content.append({"type": "image", "image": img})

    messages: List[dict] = [{"role": "user", "content": user_content}]
    if output_text is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": output_text}],
            }
        )
    return messages, images


def _get_base_model_name_from_lora(
    lora_module_list: List[str],
    model_name_or_path: Optional[str] = None,
) -> str:
    """
    从 LoRA 目录的 adapter_config.json 中解析出 base_model_name_or_path。
    若用户显式传入 model_name_or_path，则直接使用。
    """
    if model_name_or_path is not None:
        return model_name_or_path
    if not lora_module_list:
        raise ValueError("lora_module_list 不能为空")

    default_peft_model_id = lora_module_list[0]
    adapter_cfg_path = os.path.join(default_peft_model_id, "adapter_config.json")
    if not os.path.exists(adapter_cfg_path):
        raise FileNotFoundError(f"adapter_config.json not found in {default_peft_model_id}")
    import json

    with open(adapter_cfg_path, "r", encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    base_model_name = adapter_cfg.get("base_model_name_or_path", None)
    if base_model_name is None:
        raise ValueError(
            f"'base_model_name_or_path' not found in {adapter_cfg_path}, "
            "please specify model_name_or_path explicitly."
        )
    return base_model_name


def _mm_get_loss(
    example_inputs: List[str],
    example_outputs: List[str],
    model: Qwen2VLForConditionalGeneration,
    processor: AutoProcessor,
    batch_size: Optional[int],
) -> float:
    """
    使用 Qwen2‑VL 官方推荐的 processor / process_vision_info 计算 few-shot loss。

    - 将 (text, images) 组装成多模态对话 messages
    - 通过 processor 得到 input_ids + image_inputs
    - 在 model(**inputs, labels=...) 中同时传入 images / videos 等信息
    """
    if batch_size is None or batch_size <= 0:
        batch_size = len(example_inputs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_examples = 0

    for start in range(0, len(example_inputs), batch_size):
        batch_inp = example_inputs[start : start + batch_size]
        batch_out = example_outputs[start : start + batch_size]

        # 对 batch 中每个样本分别构造 messages / vision info / chat_template 文本
        texts: List[str] = []
        batch_image_inputs = []
        for inp, out in zip(batch_inp, batch_out):
            messages, _ = _build_mm_messages_and_images(inp, out)
            # 训练阶段不需要 generation prompt
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            image_inputs, _ = process_vision_info(messages)
            texts.append(text)
            batch_image_inputs.append(image_inputs)

        # 这里只有图像，没有视频，显式不传 videos，避免空视频列表触发内部错误
        inputs = processor(
            text=texts,
            images=batch_image_inputs,
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 官方推荐：labels 为 input_ids 的拷贝，并对 padding 位置 mask
        labels = inputs["input_ids"].clone()
        if "attention_mask" in inputs:
            labels[inputs["attention_mask"] == 0] = -100

        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss.detach().float()

        bs = len(batch_inp)
        total_loss += float(loss) * bs
        total_examples += bs

    if total_examples == 0:
        return 0.0
    return total_loss / total_examples


def _mm_get_score(
    weights,
    model: Qwen2VLForConditionalGeneration,
    cache,
    lora_module_list: List[str],
    example_inputs: List[str],
    example_outputs: List[str],
    batch_size: Optional[int],
    processor: AutoProcessor,
    get_regular,
):
    """
    多模态版本的 get_score：
    - 用给定的 weights 线性组合各个 LoRA 的权重
    - 使用 Qwen2‑VL 的多模态 loss 作为优化目标（加上正则项）
    """
    # 线性组合 LoRA 权重
    final_state_dict = get_final_weights(weights, lora_module_list, cache)
    set_peft_model_state_dict(model, final_state_dict)

    # 计算 few-shot loss
    loss = _mm_get_loss(
        example_inputs=example_inputs,
        example_outputs=example_outputs,
        model=model,
        processor=processor,
        batch_size=batch_size,
    )
    metric_val = loss + get_regular(weights)
    return metric_val


def lorahub_learning(
    lora_module_list: List[str],
    example_inputs: List[str],
    example_outputs: List[str],
    max_inference_step: int,
    model_name_or_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    get_regular=default_l1_regularization,
    seed: int = 42,
):
    """
    Qwen2‑VL 多模态版本的 LoRAHub 学习过程：

    - 使用 swift 的 load_base_model_and_lora_modules 加载带 LoRA 结构的 Qwen2‑VL 模型
    - 额外加载 AutoProcessor 和 qwen_vl_utils.process_vision_info 以支持图片像素输入
    - 在 few-shot loss 中同时考虑文本和图片
    """
    _set_seed(seed)

    number_of_loras = len(lora_module_list)
    if number_of_loras == 0:
        print("> No LoRA modules are provided. Please provide at least one LoRA module.")
        return None, None, None

    # 解析底模 id，并加载多模态 processor
    base_model_name = _get_base_model_name_from_lora(
        lora_module_list, model_name_or_path=model_name_or_path
    )
    processor = AutoProcessor.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )

    # 使用原有的加载逻辑拿到带 LoRA 的 Qwen2‑VL 模型
    model, _tokenizer, cache = load_base_model_and_lora_modules(
        lora_module_list, model_name_or_path=base_model_name
    )

    # Nevergrad 的优化器配置与原始实现保持一致
    instrum = ng.p.Array(
        init=[0.0] * number_of_loras,
        upper=[1.5] * number_of_loras,
        lower=[-1.5] * number_of_loras,
    )
    optimizer = ng.optimizers.NGOpt(
        parametrization=instrum,
        budget=max_inference_step,
    )

    print("> Begin to perform gradient-free optimization (multimodal, Qwen2‑VL) ...")

    get_score_partial = lambda w: _mm_get_score(
        weights=w,
        model=model,
        cache=cache,
        lora_module_list=lora_module_list,
        example_inputs=example_inputs,
        example_outputs=example_outputs,
        batch_size=batch_size,
        processor=processor,
        get_regular=get_regular,
    )

    recommendation = optimizer.minimize(get_score_partial, verbosity=1)

    final_lora = get_final_weights(recommendation.value, lora_module_list, cache)
    set_peft_model_state_dict(model, final_lora)

    # 将 LoRA 合并回底模，得到一个可直接用于推理的 Qwen2‑VL 模型
    merged_model = model.merge_and_unload()

    return recommendation.value, merged_model, processor


def lorahub_inference(
    example_inputs: List[str],
    model_or_name_path: Union[Any, str],
    tokenizer_or_tokenizer_path: Union[AutoProcessor, str],
    batch_size: int,
    example_outputs: Optional[List[str]] = None,
):
    """
    使用 Qwen2‑VL 的 AutoProcessor + process_vision_info 进行多模态推理。

    - example_inputs: 含有 [IMAGE_PATHS] 块的输入文本
    - model_or_name_path: 已 merge 的 Qwen2‑VL 模型或其路径
    - tokenizer_or_tokenizer_path: AutoProcessor 或其路径（保持原接口命名以兼容旧代码）
    """

    def _simple_tokenize(text: str):
        return text.lower().strip().split()

    def _tfidf_cosine(a: str, b: str) -> float:
        tokens_a = _simple_tokenize(a)
        tokens_b = _simple_tokenize(b)
        if not tokens_a or not tokens_b:
            return 0.0

        vocab = {}
        for tok in set(tokens_a + tokens_b):
            vocab.setdefault(tok, len(vocab))

        import math

        def build_vec(tokens):
            tf = [0.0] * len(vocab)
            for t in tokens:
                if t in vocab:
                    tf[vocab[t]] += 1.0
            return tf

        tf_a = build_vec(tokens_a)
        tf_b = build_vec(tokens_b)

        df = [0] * len(vocab)
        for i, v in enumerate(tf_a):
            if v > 0:
                df[i] += 1
        for i, v in enumerate(tf_b):
            if v > 0:
                df[i] += 1
        N = 2.0
        idf = [math.log((N + 1.0) / (d + 1.0)) + 1.0 for d in df]

        vec_a = [tf_a[i] * idf[i] for i in range(len(vocab))]
        vec_b = [tf_b[i] * idf[i] for i in range(len(vocab))]

        dot = sum(x * y for x, y in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(x * x for x in vec_a))
        norm_b = math.sqrt(sum(x * x for x in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def accuracy_score(outputs, ground_truths):
        correct = 0
        total = 0
        for output, truth in zip(outputs, ground_truths):
            sim = _tfidf_cosine(output, truth)
            if sim > 0.6:
                correct += 1
            total += 1
        if total == 0:
            return 0.0
        return correct / total * 100.0

    # 加载模型
    if isinstance(model_or_name_path, str):
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_or_name_path,
            trust_remote_code=True,
        )
    else:
        model = model_or_name_path

    # 加载 / 兼容 AutoProcessor
    if isinstance(tokenizer_or_tokenizer_path, str):
        processor = AutoProcessor.from_pretrained(
            tokenizer_or_tokenizer_path,
            trust_remote_code=True,
        )
    else:
        processor = tokenizer_or_tokenizer_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    example_predictions: List[str] = []

    for start in range(0, len(example_inputs), batch_size):
        batch_inp = example_inputs[start : start + batch_size]

        texts: List[str] = []
        batch_image_inputs = []

        for inp in batch_inp:
            # 推理时只包含 user turn，按照官方推荐构造 messages
            user_text, image_paths = _parse_text_and_image_paths(inp or "")
            images = _load_images(image_paths)

            user_content = [{"type": "text", "text": user_text}]
            for img in images:
                user_content.append({"type": "image", "image": img})

            messages = [{"role": "user", "content": user_content}]

            # 推理阶段需要 generation prompt
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)

            texts.append(text)
            batch_image_inputs.append(image_inputs)

        # 同样这里只处理图片，多模态视频为空时不传 videos，避免内部 video_utils 报错
        inputs = processor(
            text=texts,
            images=batch_image_inputs,
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
            )

        # AutoProcessor 会把解码委托给内部 tokenizer
        try:
            outputs = processor.batch_decode(generated_ids, skip_special_tokens=True)
        except AttributeError:
            outputs = processor.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
        example_predictions.extend(outputs)

    if example_outputs is not None:
        task_perf = accuracy_score(example_predictions, example_outputs)
    else:
        task_perf = None

    return example_predictions, task_perf


