import os
import random
from multiprocessing import Pool, cpu_count
import json


def transform_jsonl_file(file_path, output_folder):
    """格式化字段"""
    output_file_path = os.path.join(output_folder, os.path.basename(file_path))
    output_lines = []

    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            output_lines.append({
                    'dataset': 'matrix',
                    'source_file': file_path,
                    'text': data['text'] if 'text' in data else data['content'] 
                })

    # 将格式化结果写入输出文件
    with open(output_file_path, 'w') as f:
        # f.writelines(output_lines)
        for line in output_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

def process_file(args):
    """处理单个文件"""
    file_path, output_folder = args
    print(f"processing {file_path}...")
    transform_jsonl_file(file_path, output_folder)
    print(f"{file_path} processing done.")




def main(input_folder, output_folder):
    """主函数，负责并行处理文件"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有 JSONL 文件
    jsonl_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.jsonl')]

    # 使用多进程并行处理
    with Pool(cpu_count()//2) as pool:
        pool.map(process_file, [(file, output_folder) for file in jsonl_files])

if __name__ == "__main__":
    input_folder = './data/tokenizers/matrix_sampled'
    output_folder = './data/tokenizers/matrix_sampled_mapped'
    
    main(input_folder, output_folder)