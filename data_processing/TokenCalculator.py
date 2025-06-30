# pip3 install transformers matplotlib numpy
# python3 TokenCalculator.py
import json
from matplotlib.font_manager import FontProperties
from tqdm import tqdm
import transformers
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
from pathlib import Path
from typing import List, Dict, Union, Tuple, Any, Optional

from utils.misc_utils import get_data_dir, get_project_root

chat_tokenizer_dir = get_project_root() / "data_models/tokenizers/deepseek/"
qwen_chat_tokenizer_dir = get_project_root() / "data_models/tokenizers/qwen/"


class TokenCalculator:
    """Token计算器，用于统计训练数据的token数量及分布情况"""
    
    def __init__(self, tokenizer_path: Union[str, Path, None] = None, trust_remote_code: bool = True):
        """初始化Token计算器
        
        Args:
            tokenizer_path: tokenizer路径，默认使用qwen tokenizer
            trust_remote_code: 是否信任远程代码
        """
        if tokenizer_path is None:
            tokenizer_path = qwen_chat_tokenizer_dir
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=trust_remote_code
        )
        
    def load_data(self, file_path: Union[str, Path]) -> List[Dict]:
        """加载数据

        Args:
            file_path: 数据文件路径

        Returns:
            数据列表
        """
        # ensure file_path is string for suffix checks
        file_path_str = str(file_path)
        raw_data = []

        try:
            if file_path_str.endswith(".json"):
                # 如果是JSON文件，直接加载
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    if isinstance(data, list):
                        raw_data = data
                    else:
                        raw_data = [data]
            elif file_path_str.endswith(".jsonl"):
                # 如果是JSONL文件，逐行加载
                with open(file_path, "r", encoding="utf-8") as file:
                    for line in file:
                        # 移除行尾的换行符
                        cleaned_line = line.strip()
                        if cleaned_line:  # 确保非空行
                            # 解析JSON
                            data = json.loads(cleaned_line)
                            raw_data.append(data)
            else:
                raise ValueError(f"不支持的文件格式: {file_path}")
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return []
            
        print(f"成功加载 {len(raw_data)} 条数据")
        return raw_data

    def calculate_tokens(self, data: List[Dict]) -> Tuple[List[int], int]:
        """计算每条数据的token数量
        
        Args:
            data: 数据列表
            
        Returns:
            每条数据的token数量列表和总token数
        """
        token_counts = []
        total_tokens = 0

        for item in tqdm(data):
            # 直接使用json.dumps将数据项转换为字符串
            text = json.dumps(item, ensure_ascii=False)
            tokens = self.tokenizer.encode(text)
            token_count = len(tokens)
            token_counts.append(token_count)
            total_tokens += token_count
            
        return token_counts, total_tokens
    
    def calculate_statistics(self, token_counts: List[int]) -> Dict[str, Union[float, int]]:
        """计算统计信息
        
        Args:
            token_counts: token数量列表
            
        Returns:
            统计信息字典
        """
        if not token_counts:
            return {
                "count": 0,
                "mean": 0,
                "median": 0,
                "std": 0,
                "min": 0,
                "max": 0,
                "quartile_25": 0,
                "quartile_75": 0
            }
            
        np_counts = np.array(token_counts)
        
        return {
            "count": len(token_counts),
            "mean": float(np.mean(np_counts)),
            "median": float(np.median(np_counts)),
            "std": float(np.std(np_counts)),
            "min": int(np.min(np_counts)),
            "max": int(np.max(np_counts)),
            "quartile_25": float(np.percentile(np_counts, 25)),
            "quartile_75": float(np.percentile(np_counts, 75))
        }
    
    def plot_histogram(self, token_counts: List[int], title: str = "Token分布直方图", 
                      save_path: Union[str, Path, None] = None, show: bool = True) -> None:
        """绘制直方图
        
        Args:
            token_counts: token数量列表
            title: 图表标题
            save_path: 保存路径，如果为None则不保存
            show: 是否显示图表
        """
        if not token_counts:
            print("没有数据可供绘图")
            return
            # 尝试使用系统中已安装的中文字体
        try:
            font_path = get_project_root() / "utils/simhei.ttf"
            chinese_font = FontProperties(fname=str(font_path))
        except Exception:
            print("警告：无法加载中文字体，将使用系统默认字体")
            chinese_font = FontProperties()
            
        plt.figure(figsize=(10, 6))
        plt.hist(token_counts, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        
        # 添加均值和中位数的垂直线
        mean = np.mean(token_counts)
        median = np.median(token_counts)

        plt.axvline(float(mean), color='red', linestyle='dashed', linewidth=1, label=f'Average: {mean:.2f}')
        plt.axvline(float(median), color='green', linestyle='dashed', linewidth=1, label=f'Median: {median:.2f}')

        plt.title(title, fontproperties=chinese_font)
        plt.xlabel('Token数量', fontproperties=chinese_font)
        plt.ylabel('数据条数', fontproperties=chinese_font)
        plt.grid(axis='y', alpha=0.75)
        plt.legend()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
            
        if show:
            plt.show()
        else:
            plt.close()
    
    def save_statistics_to_file(self, token_counts: List[int], statistics: Dict, 
                              file_path: Union[str, Path]) -> None:
        """将统计结果保存到文件
        
        Args:
            token_counts: token数量列表
            statistics: 统计信息
            file_path: 文件保存路径
        """
        try:
            result = {
                "timestamp": datetime.datetime.now().isoformat(),
                "statistics": statistics,
                "token_counts": token_counts
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
            print(f"统计结果已保存至: {file_path}")
        except Exception as e:
            print(f"保存统计结果时出错: {e}")
    
    def analyze_file(self, file_path: Union[str, Path], save_dir: Union[str, Path, None] = None, 
                   save_prefix: Optional[str] = None, show_plot: bool = True) -> Dict:
        """分析文件的token分布情况
        
        Args:
            file_path: 数据文件路径
            save_dir: 结果保存目录，默认为数据文件所在目录
            save_prefix: 保存文件前缀，默认为数据文件名
            show_plot: 是否显示图表
            
        Returns:
            分析结果字典
        """
        # 处理路径
        file_path = Path(file_path)
        if save_dir is None:
            save_dir = file_path.parent
        else:
            save_dir = Path(save_dir)
            
        if save_prefix is None:
            save_prefix = file_path.stem
            
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 加载数据
        data = self.load_data(file_path)
        if not data:
            print(f"无法加载数据或数据为空: {file_path}")
            return {}
            
        # 计算token
        token_counts, total_tokens = self.calculate_tokens(data)
        
        # 计算统计信息
        statistics = self.calculate_statistics(token_counts)
        
        # 打印统计信息
        print(f"\n{'='*50}")
        print(f"文件: {file_path}")
        print(f"总数据条数: {statistics['count']}")
        print(f"总Token数: {total_tokens}")
        print(f"每条数据平均Token数: {statistics['mean']:.2f}")
        print(f"Token数中位数: {statistics['median']:.2f}")
        print(f"Token数标准差: {statistics['std']:.2f}")
        print(f"最小Token数: {statistics['min']}")
        print(f"最大Token数: {statistics['max']}")
        print(f"25%分位数: {statistics['quartile_25']:.2f}")
        print(f"75%分位数: {statistics['quartile_75']:.2f}")
        print(f"{'='*50}\n")
        
        # 保存统计结果
        stats_file = save_dir / f"{save_prefix}_token_stats.json"
        self.save_statistics_to_file(token_counts, statistics, stats_file)
        
        # 绘制直方图
        plot_file = save_dir / f"{save_prefix}_token_histogram.png"
        self.plot_histogram(
            token_counts, 
            title=f"Token分布直方图 - {file_path.name}",
            save_path=plot_file,
            show=show_plot
        )
        
        return {
            "file_path": str(file_path),
            "total_tokens": total_tokens,
            "statistics": statistics,
            "stats_file": str(stats_file),
            "plot_file": str(plot_file)
        }


# 示例用法
if __name__ == "__main__":
    # 初始化Token计算器
    calculator = TokenCalculator()
    
    # 分析指定数据文件
    data_path = get_data_dir() / "training_data_sharegpt_gemini-2.5-pro_20250629_103625_30_v3.jsonl"
    calculator.analyze_file(data_path)
