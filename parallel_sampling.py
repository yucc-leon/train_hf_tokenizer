import os
import random
from multiprocessing import Pool, cpu_count


def sample_jsonl_file(file_path, sample_ratio, output_folder):
    """从 JSONL 文件中抽样出一定比例的数据并保存到输出文件夹"""
    output_file_path = os.path.join(output_folder, os.path.basename(file_path))
    total_lines = 0
    sampled_lines = []

    # 首先通过两次遍历文件来计算总行数
    with open(file_path, 'r') as f:
        for _ in f:
            total_lines += 1

    sample_size = int(total_lines * sample_ratio)

    # 使用 reservoir sampling 算法逐行读取并抽样
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if len(sampled_lines) < sample_size:
                sampled_lines.append(line)
            else:
                replace_index = random.randint(0, line_num - 1)
                if replace_index < sample_size:
                    sampled_lines[replace_index] = line

    # 将抽样结果写入输出文件
    with open(output_file_path, 'w') as f:
        f.writelines(sampled_lines)

def process_file(args):
    """处理单个文件"""
    file_path, sample_ratio, output_folder = args
    print(f"processing {file_path}...")
    sample_jsonl_file(file_path, sample_ratio, output_folder)
    print(f"{file_path} processing done.")




def main(input_folder, output_folder, sample_ratio=0.1):
    """主函数，负责并行处理文件"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有 JSONL 文件
    jsonl_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.jsonl')]

    # 使用多进程并行处理
    with Pool(cpu_count()//2) as pool:
        pool.map(process_file, [(file, sample_ratio, output_folder) for file in jsonl_files])

if __name__ == "__main__":
    input_folder = './dataset/m-a-p/Matrix'
    output_folder = './data/raw_texts_for_training/'
    sample_ratio = 1/700  # 抽样比例，可以根据需要调整

    main(input_folder, output_folder, sample_ratio)