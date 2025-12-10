import json
from typing import List, Dict

from lorahub.algorithm_qwen2vl_mm import (
    lorahub_learning,
    lorahub_inference,
    build_model_with_fixed_weights,
)

# === 配置区 ===

# 1）LoRA 模块的路径（建议用绝对路径）
LORA_MODULES = [
    "/home/hmpiao/hmpiao/xuerong/FedMABench/output/category_lora_entertainment",
    "/home/hmpiao/hmpiao/xuerong/FedMABench/output/category_lora_office",
    "/home/hmpiao/hmpiao/xuerong/FedMABench/output/category_lora_shopping",
    "/home/hmpiao/hmpiao/xuerong/FedMABench/output/category_lora_traveling",
    "/home/hmpiao/hmpiao/xuerong/FedMABench/output/category_lora_lives",
]

# （可选）固定好的 LoRA 加权向量，用于跳过搜索、加速 debug
# 长度必须与 LORA_MODULES 相同
USE_FIXED_WEIGHTS = True
FIXED_WEIGHTS = [0.12282166, -0.00815929, -0.01167593, -0.00863875, -0.01619601]

# 2）few-shot 学习用的数据（example.jsonl）和推理评估用的数据（infer.jsonl）
EXAMPLE_JSONL = "example.jsonl"
INFER_JSONL = "infer.jsonl"

# 3）图片路径前缀重写（与 swift 的 --image_prefix_src/dst 类似）
#    如果不需要重写，可以把两个常量都设为 ""。
IMAGE_PREFIX_SRC = "/ailab/user/wangwenhao/ms-swift/androidcontrol_1108/unpack-androidcontrol/"
IMAGE_PREFIX_DST = "./android_control_unpack/"


def _rewrite_image_path(path: str) -> str:
    """按照 IMAGE_PREFIX_SRC/IMAGE_PREFIX_DST 重写图片路径"""
    if IMAGE_PREFIX_SRC and IMAGE_PREFIX_DST and path.startswith(IMAGE_PREFIX_SRC):
        return IMAGE_PREFIX_DST + path[len(IMAGE_PREFIX_SRC):]
    return path


def load_examples_from_example_jsonl(
    path: str, max_examples: int = 5
) -> List[Dict[str, str]]:
    """
    从 example.jsonl 中读取若干条样本，转成 LoraHub 需要的 input / output 文本。

    约定：
    - input:  用户 turn 的文本 + 图片路径列表（作为文本附在后面）
    - output: assistant turn 的动作序列文本

    注意：当前 LoraHub 是纯文本接口，这里只是把图片路径编码进文本中，
    真正的像素信息不会进入模型；要真正做多模态，需要改 lorahub.algorithm 本身。
    """
    examples: List[Dict[str, str]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)

            convs = data["conversations"]
            user_turn = next(c for c in convs if c["from"] == "user")
            assistant_turn = next(c for c in convs if c["from"] == "assistant")
            images = data.get("images", [])

            # 把图片路径也拼到 prompt 里，方便之后如果要接入 VLM，可以在这里再做解析
            images_block = ""
            if images:
                rewritten = [_rewrite_image_path(p) for p in images]
                images_block = "\n\n[IMAGE_PATHS]\n" + "\n".join(rewritten)

            input_text = user_turn["value"].strip() + images_block
            output_text = assistant_turn["value"].strip()

            examples.append({"input": input_text, "output": output_text})

            if len(examples) >= max_examples:
                break

    return examples


def load_inputs_and_labels_from_infer_jsonl(
    path: str, max_examples: int = 50
) -> (List[str], List[str]):
    """
    从 infer.jsonl 中读取若干条样本，构造推理用的输入字符串和标签字符串。

    约定：
    - input: 使用 instruction 作为主要文本，并把 imgs 中的图片路径附在末尾（仅作为占位）
    - label: 将 acts_convert 列表按行拼接成一个字符串，作为目标动作序列
    """
    inputs: List[str] = []
    labels: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)

            instruction = data["instruction"].strip()
            imgs = data.get("imgs", [])
            acts_convert = data.get("acts_convert", [])

            # 构造输入：指令 + 图片路径占位
            images_block = ""
            if imgs:
                rewritten = [_rewrite_image_path(p) for p in imgs]
                images_block = "\n\n[IMAGE_PATHS]\n" + "\n".join(rewritten)
            input_text = instruction + images_block
            inputs.append(input_text)

            # 构造标签：把动作序列按行拼接
            if isinstance(acts_convert, list):
                label_text = "\n".join(acts_convert).strip()
            else:
                # 如果格式异常，退化为空字符串，避免报错
                label_text = ""
            labels.append(label_text)

            if len(inputs) >= max_examples:
                break

    return inputs, labels


# few-shot 学习数据：直接从 example.jsonl 读取前 N 条
def get_examples_for_learning(n_examples: int = 5) -> List[Dict[str, str]]:
    return load_examples_from_example_jsonl(EXAMPLE_JSONL, max_examples=n_examples)


def main():
    # 构造 input / output 列表
    examples = get_examples_for_learning(n_examples=5)
    example_inputs = [e["input"] for e in examples]
    example_outputs = [e["output"] for e in examples]

    # 如果 LoRA 的 config 里已经包含 base_model_name，就可以把 model_name_or_path 留空；
    # 如果你自己有一个本地底模目录，比如 base_model，就这样写：
    base_model_path = None  # 如果有单独的底模目录，可以改成对应路径

    if USE_FIXED_WEIGHTS:
        # 使用已经学习好的固定权重，直接构建合并后的模型，跳过 Nevergrad 搜索
        module_weights = FIXED_WEIGHTS
        print("use fixed weights:", module_weights)
        model, tokenizer = build_model_with_fixed_weights(
            lora_module_list=LORA_MODULES,
            weights=module_weights,
            model_name_or_path=base_model_path,
        )
    else:
        module_weights, model, tokenizer = lorahub_learning(
            lora_module_list=LORA_MODULES,
            example_inputs=example_inputs,
            example_outputs=example_outputs,
            max_inference_step=20,  # 搜索步数，可以先用 20/40 试
            batch_size=1,
            model_name_or_path=base_model_path,  # 或者 None 让它从 LoRA config 里自动读
        )

        print("learned weights:", module_weights)

    # 用组合后的模型在 infer.jsonl 上做推理
    test_inputs, test_labels = load_inputs_and_labels_from_infer_jsonl(
        INFER_JSONL, max_examples=100
    )
    preds, acc = lorahub_inference(
        example_inputs=test_inputs,
        model_or_name_path=model,  # 直接传上面返回的合并后模型
        tokenizer_or_tokenizer_path=tokenizer,
        batch_size=8,
        example_outputs=test_labels,  # 使用 acts_convert 作为标签，计算 accuracy
    )
    print("num_test_examples:", len(test_inputs))
    print("predictions_example_0_3:", preds[:3])
    print("accuracy:", acc)


if __name__ == "__main__":
    main()