import os
from tqdm import tqdm
import json
from eval_gpt import calculate_tfidf
import argparse


def read_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def judge_step(a, b):
    if calculate_tfidf(a, b) > 0.6:
        return 1
    else:
        return 0


def calculate_step_accuracy(data):
    """
    计算每一行的准确率，判断每一行的 label 和 response 是否一致。
    返回每行的准确性列表 [1, 0, 1, ...]。
    """
    step_accuracies = []
    for item in data:
        # 如果 label 和 response 相同，认为是相关的
        if item['label'] != 'Click at a button':
            step_accuracies.append(judge_step(item['label'], item['response']))

    return step_accuracies


def calculate_episode_accuracy(data, step_accuracies=None):
    """
    按照 query 对数据进行分组，然后计算每个 episode 的准确率。
    如果一个 episode 中所有行的 label 和 response 都相同，准确率为 1，否则为 0。
    """


    episode_accuracies = []
    episodes = {}
    import re


    # 按照 query 对数据进行分组
    for item in data:
        query = item['query']
        original_query = query
        
        # 尝试提取 User Instruction（如果格式是 ### User Instruction ###\n...\n###）
        match = re.search(r'### User Instruction ###\n(.*?)\n###', query, re.DOTALL)
        if match:
            instruction = match.group(1).strip()  # 提取 User Instruction 部分并去除首尾空格
        else:
            # 如果没有找到标准格式，尝试提取 <image> 标签后的指令部分
            # 格式通常是: <image>\n<image>\n...\n指令内容
            lines = query.split('\n')
            # 找到第一个不是 <image> 的行，作为指令开始
            instruction_lines = []
            for line in lines:
                line_stripped = line.strip()
                if line_stripped == '<image>' or line_stripped == '':
                    continue
                # 找到第一个非空且不是 <image> 的行，后面的都是指令
                instruction_lines.append(line)
            
            if instruction_lines:
                # 合并所有指令行
                instruction = '\n'.join(instruction_lines).strip()
            else:
                # 如果还是找不到，就使用原始 query（去掉开头的 <image> 标签）
                instruction = '\n'.join([line for line in lines if line.strip() != '<image>']).strip()
        
        # 使用提取的instruction作为分组键
        if instruction not in episodes:
            episodes[instruction] = []
        episodes[instruction].append(item)

    # 对每个 episode 检查所有行是否相关
    for instruction, episode_items in episodes.items():
        # 如果 episode 中所有行的准确性都为 1，则认为该 episode 的准确率为 1
        if all(judge_step(item['label'], item['response']) for item in episode_items):
            episode_accuracies.append(1)
        else:
            episode_accuracies.append(0)
    
    # 调试信息：显示episode分组情况和提取的instruction示例
    print(f"\n调试信息:")
    print(f"  总样本数: {len(data)}")
    print(f"  唯一instruction数: {len(episodes)}")
    episode_sizes = {}
    for instruction, items in episodes.items():
        size = len(items)
        episode_sizes[size] = episode_sizes.get(size, 0) + 1
    print(f"  Episode大小分布: {episode_sizes}")
    
    # 显示前3个提取的instruction示例
    print(f"\n  提取的User Instruction示例（前3个）:")
    for i, (instruction, items) in enumerate(list(episodes.items())[:3], 1):
        print(f"    {i}. {instruction[:80]}... (对应 {len(items)} 个样本)")
    
    if len(episodes) == len(data):
        print(f"\n  注意: 每个instruction只对应1个样本")
        print(f"  这会导致step-level和episode-level准确率相同，因为：")
        print(f"    - Step-level: 比较每个样本的response和label")
        print(f"    - Episode-level: 每个episode只有1个样本，所以结果相同")

    return episode_accuracies


def test_main(data_path):
    # 读取数据
    # data_path = r'/ailab/user/wangwenhao/FedMobile/output/qwen2-vl-7b-instruct/v7-20241219-094924/global_lora_49/infer_result/20241220-001128.jsonl'
    data = read_jsonl(data_path)
    
    # 获取数据所在目录和文件名
    output_dir = os.path.dirname(data_path)
    base_filename = os.path.splitext(os.path.basename(data_path))[0]  # 获取不带后缀的文件名
    output_file = os.path.join(output_dir, base_filename + '.log')    # 构建 .log 文件路径

    # 打开文件保存输出

    # 计算每一行的准确率
    step_accuracies = calculate_step_accuracy(data)
    step_accuracy = sum(step_accuracies) / len(step_accuracies)
    print(f"Step-level accuracy: {step_accuracy * 100:.2f}%")

    # 计算按 query 分组的 episode 准确率
    episode_accuracies = calculate_episode_accuracy(data)
    episode_accuracy = sum(episode_accuracies) / len(episode_accuracies)
    print(f"len: {len(episode_accuracies)}")
    print(f"Episode-level accuracy: {episode_accuracy * 100:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="")
    # parser.add_argument("--save_failed_generation", action='store_true', default=False)

    args = parser.parse_args()

    test_main(args.data_path)
