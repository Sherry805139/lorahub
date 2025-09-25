from lorahub.algorithm import lorahub_inference
import os
import json
from lorahub.algorithm import lorahub_learning, lorahub_inference
from lorahub.constant import LORA_MODULE_NAMES
import random
from random import shuffle
from typing import Dict, List, Tuple


def evaluate_flan_results_zero_shot(folder: str, flan_model_name: str) -> Tuple[Dict[str, float], float]:
    """评估 FLAN 模型在每个任务上的 zero-shot 表现。

    返回 (task->acc 字典, 平均 acc)。
    """
    sub_dirs = sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])

    task_to_acc: Dict[str, float] = {}
    task_acc_list: List[float] = []

    for sub_dir in sub_dirs:
        test_file_path = os.path.join(folder, sub_dir, "zero_shot.jsonl")
        task_inputs, task_outputs = [], []
        for line in open(test_file_path, "r", encoding="utf-8"):
            example = json.loads(line)
            task_inputs.append(example["context"])
            task_outputs.append(example["completion"])
        print("Evaluating on task (zero shot): ", sub_dir)
        _, task_acc = lorahub_inference(
            task_inputs,
            flan_model_name,
            flan_model_name,
            16,
            task_outputs,
        )
        print("acc:", task_acc)
        task_acc_list.append(task_acc)
        task_to_acc[sub_dir] = task_acc
    avg_acc = sum(task_acc_list) / len(task_acc_list) if task_acc_list else 0.0
    print("average acc:", avg_acc)
    return task_to_acc, avg_acc


def evaluate_flan_results_few_shot(folder: str, flan_model_name: str) -> Tuple[Dict[str, float], float]:
    """评估 FLAN 模型在每个任务上的 few-shot (five-shot) 表现。

    返回 (task->acc 字典, 平均 acc)。
    """
    sub_dirs = sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])

    task_to_acc: Dict[str, float] = {}
    task_acc_list: List[float] = []

    for sub_dir in sub_dirs:
        test_file_path = os.path.join(folder, sub_dir, "few_shot.jsonl")
        task_inputs, task_outputs = [], []
        for line in open(test_file_path, "r", encoding="utf-8"):
            example = json.loads(line)
            task_inputs.append(example["context"])
            task_outputs.append(example["completion"])
        print("Evaluating on task (five shot): ", sub_dir)
        _, task_acc = lorahub_inference(
            task_inputs,
            flan_model_name,
            flan_model_name,
            16,
            task_outputs,
        )
        print("acc:", task_acc)
        task_acc_list.append(task_acc)
        task_to_acc[sub_dir] = task_acc
    avg_acc = sum(task_acc_list) / len(task_acc_list) if task_acc_list else 0.0
    print("average acc:", avg_acc)
    return task_to_acc, avg_acc

def evaluate_lorahub_results_few_shot(folder: str) -> Tuple[Dict[str, float], float]:
    """对每个任务：用 5 个示例进行 LoRAHub 学习，并在 zero_shot 集上评估。

    返回 (task->平均 acc(5 seeds), 平均 acc across tasks)。
    """
    sub_dirs = sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])

    task_to_perf: Dict[str, float] = {}
    all_task_perfs: List[float] = []

    # 就是把data_bbh的数据读出来，然后随机选5个例子，然后训练一个LoRA模型
    # 然后评估这个LoRA模型在zero_shot.jsonl上的表现
    # 然后对 5 个随机种子取平均
    for sub_dir in sub_dirs:
        # construct the few-shot examples for lorahub learning
        example_inputs, examples_outputs = [], []
        example_file_path = os.path.join(folder, sub_dir, "example.jsonl")
        for line in open(example_file_path, "r", encoding="utf-8"):
            example = json.loads(line)
            example_inputs.append(example["context"])
            examples_outputs.append(example["completion"])

        # random select 5 examples for each task，random.seed(42) 是随机种子
        random.seed(42)
        shuffled_set = list(zip(example_inputs, examples_outputs))
        random.shuffle(shuffled_set)
        example_inputs, examples_outputs = zip(*shuffled_set)
        # take the first 5 examples，这里就是取前5个例子
        example_inputs, examples_outputs = list(example_inputs[:5]), list(examples_outputs[:5])

        # load the zero-shot examples for evaluation，这里就是读取zero_shot.jsonl的数据
        test_file_path = os.path.join(folder, sub_dir, "zero_shot.jsonl")
        task_inputs, task_outputs = [], []
        for line in open(test_file_path, "r", encoding="utf-8"):
            example = json.loads(line)
            task_inputs.append(example["context"])
            task_outputs.append(example["completion"])

        task_perf_list: List[float] = []
        for seed in range(1, 6):
            random.seed(seed)

            def get_lora_module_list() -> List[str]:
                return random.sample(LORA_MODULE_NAMES, 20)
            # get a list of modules to be used in the composition
            modules = get_lora_module_list()

            # perform LoRAHub learning
            module_weights, model, tokenizer = lorahub_learning(
                lora_module_list=modules,
                example_inputs=example_inputs,
                example_outputs=examples_outputs,
                max_inference_step=40,
                batch_size=5,
            )

            print("module_weights:", module_weights)

            # Perform inference to get predictions
            _, task_acc = lorahub_inference(
                example_inputs=task_inputs,
                model_or_name_path=model,
                tokenizer_or_tokenizer_path=tokenizer,
                batch_size=10,
                # can set as None if you do not have the ground truth
                example_outputs=task_outputs,
            )
            task_perf_list.append(task_acc)

        avg_perf = sum(task_perf_list) / len(task_perf_list) if task_perf_list else 0.0
        max_perf = max(task_perf_list) if task_perf_list else 0.0
        print("Task:", sub_dir, "average perf:", avg_perf, "best perf:", max_perf)

        task_to_perf[sub_dir] = avg_perf
        all_task_perfs.append(avg_perf)

    total_avg_perf = sum(all_task_perfs) / len(all_task_perfs) if all_task_perfs else 0.0
    print("LoRAHub overall average perf:", total_avg_perf)
    return task_to_perf, total_avg_perf


def generate_latex_table(tasks: List[str],
                         zero_dict: Dict[str, float],
                         few_dict: Dict[str, float],
                         lora_dict: Dict[str, float]) -> str:
    """根据三个结果字典生成 LaTeX 表格字符串。"""
    def fmt(x: float) -> str:
        try:
            return f"{x:.1f}"
        except Exception:
            return ""

    lines: List[str] = []
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("task & zero_{shot} & few_{shot} & lorahub \\")
    lines.append("\\midrule")
    for t in tasks:
        z = fmt(zero_dict.get(t, float("nan"))) if t in zero_dict else ""
        f = fmt(few_dict.get(t, float("nan"))) if t in few_dict else ""
        l = fmt(lora_dict.get(t, float("nan"))) if t in lora_dict else ""
        lines.append(f"{t.replace('_', '\\_')} & {z} & {f} & {l} \\\")
    # 计算平均
    def safe_avg(d: Dict[str, float]) -> float:
        vals = [v for k, v in d.items() if isinstance(v, (int, float))]
        return sum(vals) / len(vals) if vals else 0.0
    avg_z = fmt(safe_avg(zero_dict))
    avg_f = fmt(safe_avg(few_dict))
    avg_l = fmt(safe_avg(lora_dict))
    lines.append("\\midrule")
    lines.append(f"average\\_perf & {avg_z} & {avg_f} & {avg_l} \\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)

if __name__ == "__main__":
    if not os.path.exists("data_bbh"):
        # download dataset
        os.system("wget https://github.com/sail-sg/lorahub/releases/download/0.1/data_bbh.zip")
        # unzip
        os.system("unzip data_bbh.zip")

    data_root = "data_bbh"
    flan_name = "google/flan-t5-large"

    # 评估 zero-shot / few-shot（FLAN）并返回每任务结果
    zero_dict, zero_avg = evaluate_flan_results_zero_shot(data_root, flan_name)
    few_dict, few_avg = evaluate_flan_results_few_shot(data_root, flan_name)
    # 评估 LoRAHub five-shot 学习
    lora_dict, lora_avg = evaluate_lorahub_results_few_shot(data_root)

    # 统一任务顺序（27 个子文件夹）
    tasks = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])

    latex = generate_latex_table(tasks, zero_dict, few_dict, lora_dict)
    print("\n===== LaTeX Table =====\n")
    print(latex)
