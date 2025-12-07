import json
from typing import List, Dict

from lorahub.algorithm import lorahub_learning, lorahub_inference

# === 配置区 ===

# 1）LoRA 模块的路径（建议用绝对路径）
LORA_MODULES = [
    "/home/hmpiao/xuerong/FedMABench/output/category_lora_entertainment",
    "/home/hmpiao/xuerong/FedMABench/output/category_lora_office",
    "/home/hmpiao/xuerong/FedMABench/output/category_lora_shopping",
    "/home/hmpiao/xuerong/FedMABench/output/category_lora_traveling",
    "/home/hmpiao/xuerong/FedMABench/output/category_lora_lives",
]

# 2）few-shot 学习用的数据（example.jsonl）和推理评估用的数据（infer.jsonl）
EXAMPLE_JSONL = "example.jsonl"
INFER_JSONL = "infer.jsonl"


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
                images_block = "\n\n[IMAGE_PATHS]\n" + "\n".join(images)

            input_text = user_turn["value"].strip() + images_block
            output_text = assistant_turn["value"].strip()

            examples.append({"input": input_text, "output": output_text})

            if len(examples) >= max_examples:
                break

    return examples


def load_inputs_from_infer_jsonl(path: str, max_examples: int = 50) -> List[str]:
    """
    从 infer.jsonl 中读取若干条样本，构造推理用的输入字符串。

    约定：
    - 使用 instruction 作为主要文本
    - 把 imgs 中的图片路径也附在末尾，编码成文本（同上，仅作为占位）
    """
    inputs: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)

            instruction = data["instruction"].strip()
            imgs = data.get("imgs", [])

            images_block = ""
            if imgs:
                images_block = "\n\n[IMAGE_PATHS]\n" + "\n".join(imgs)

            input_text = instruction + images_block
            inputs.append(input_text)

            if len(inputs) >= max_examples:
                break

    return inputs


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

    module_weights, model, tokenizer = lorahub_learning(
        lora_module_list=LORA_MODULES,
        example_inputs=example_inputs,
        example_outputs=example_outputs,
        max_inference_step=40,  # 搜索步数，可以先用 20/40 试
        batch_size=1,
        model_name_or_path=base_model_path,  # 或者 None 让它从 LoRA config 里自动读
    )

    print("learned weights:", module_weights)

    # 用组合后的模型在 infer.jsonl 上做推理
    test_inputs = load_inputs_from_infer_jsonl(INFER_JSONL, max_examples=50)
    preds, acc = lorahub_inference(
        example_inputs=test_inputs,
        model_or_name_path=model,  # 直接传上面返回的合并后模型
        tokenizer_or_tokenizer_path=tokenizer,
        batch_size=8,
        example_outputs=None,  # infer.jsonl 没有标准答案，这里不算 accuracy
    )
    print("num_test_examples:", len(test_inputs))
    print("predictions_example_0_3:", preds[:3])
    print("accuracy:", acc)


if __name__ == "__main__":
    main()