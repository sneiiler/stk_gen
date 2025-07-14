import json
import os
import re


def convert_to_alpaca_format(input_file_path: str, output_file_path: str) -> None:
    """Reads a text file with a specific prompt format and converts it into a JSONL
    file in Alpaca format.

    Args:
        input_file_path (str): The path to the input text file.
        output_file_path (str): The path to the output JSONL file.
    """
    with open(input_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split content into individual prompts using a delimiter
    prompts = re.split(r"================ PROMPT \d+ ================", content)
    prompts = [p.strip() for p in prompts if p.strip()]

    alpaca_data = []
    for prompt_text in prompts:
        try:
            # Extract instruction
            instruction_match = re.search(
                r"(.+?)\*\*现在给你的输入数据为：\*\*", prompt_text, re.DOTALL
            )
            if not instruction_match:
                continue
            instruction = instruction_match.group(1).strip()

            # Extract input data
            input_match = re.search(
                r"\*\*现在给你的输入数据为：\*\*\s*(.+?)\s*请输出：Answer:",
                prompt_text,
                re.DOTALL,
            )
            if not input_match:
                continue
            input_data = input_match.group(1).strip()

            # Extract output
            output_match = re.search(r"请输出：Answer:(.+?)<\|EOF\|>", prompt_text, re.DOTALL)
            if not output_match:
                continue
            output_data = output_match.group(1).strip()

            alpaca_data.append(
                {
                    "instruction": instruction,
                    "input": input_data,
                    "output": output_data,
                }
            )
        except (AttributeError, IndexError) as e:
            print(f"Skipping a malformed prompt section: {e}")
            continue

    # Write to JSONL file
    with open(output_file_path, "w", encoding="utf-8") as f:
        for item in alpaca_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Successfully converted {len(alpaca_data)} prompts.")
    print(f"Output saved to: {output_file_path}")


if __name__ == "__main__":
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the project root
    project_root = os.path.dirname(current_dir)

    # Construct file paths
    input_file_name = "prompts_20250619_185329.txt"
    output_file_name = "prompts_20250619_185329.jsonl"
    
    input_file = os.path.join(project_root, "data", input_file_name)
    output_file = os.path.join(project_root, "data", output_file_name)

    # Ensure the input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
    else:
        convert_to_alpaca_format(input_file, output_file) 