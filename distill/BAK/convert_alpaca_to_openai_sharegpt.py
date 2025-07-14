import json
import uuid
from utils.misc_utils import get_data_dir

data_dir = get_data_dir()
# 输入和输出文件路径
input_path = data_dir / 'distilled_training_data_v20250618_think.jsonl'
output_path = data_dir / 'distilled_training_data_v20250618_sharegpt_format.json'
output_path_jsonl = data_dir / 'distilled_training_data_v20250618_sharegpt_format.jsonl'

def convert_alpaca_to_sharegpt(input_path, output_path):
    """
    Convert Alpaca format .jsonl file to OpenAI ShareGPT format .json file.
    将Alpaca格式的.jsonl文件批量转换为OpenAI ShareGPT格式的.json文件。

    Args:
        input_path (str): Path to the Alpaca format .jsonl file.  输入Alpaca格式的jsonl文件路径
        output_path (str): Path to save the ShareGPT format .json file.  输出ShareGPT格式json文件路径
    """
    # 读取Alpaca格式数据
    conversations = []
    with open(input_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            item = json.loads(line)
            # system角色内容为instruction
            system_value = item['instruction']
            # user输入为input字段
            user_value = item.get('input', '')
            assistant_value = item['output']
            conv = {
                "messages": [
                    {"role": "system", "content": system_value},
                    {"role": "user", "content": user_value},
                    {"role": "assistant", "content": assistant_value}
                ]
            }
            conversations.append(conv)
    # 写入ShareGPT格式
    with open(output_path, 'w', encoding='utf-8') as fout:
        json.dump(conversations, fout, ensure_ascii=False, indent=2)

def sharegpt_json_to_jsonl(json_path, jsonl_path):
    """
    Convert ShareGPT format .json file to .jsonl file (one conversation per line).
    将ShareGPT格式的json文件转换为jsonl格式，每行为一条对话。

    Args:
        json_path (str or Path): Path to the ShareGPT format .json file.  输入ShareGPT格式json文件路径
        jsonl_path (str or Path): Path to save the .jsonl file.  输出jsonl文件路径
    """
    with open(json_path, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    with open(jsonl_path, 'w', encoding='utf-8') as fout:
        for item in data:
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    # convert_alpaca_to_sharegpt(input_path, output_path)
    sharegpt_json_to_jsonl(output_path, output_path_jsonl)
