"""
数据验证器模块

该模块提供了用于验证大模型生成的卫星分簇结果的验证器类。
"""

import sys
from collections import Counter  # 在文件顶部导入
from pathlib import Path
from typing import List, Optional

# 添加项目根目录到路径
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from icecream import install

install()
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from utils.misc_utils import get_data_dir, get_project_root, get_current_timestamp
from misc_tools.sharegpt_utils import load_sharegpt_data
from data_class.sft_data_models import LLMConversationMessage

from data_class.data_validation_models import ValidationItem


class ClusterDataValidator:
    """卫星分簇结果验证器

    用于验证大模型生成的卫星分簇结果是否符合业务规则和约束条件。
    """

    def __init__(self, file_path):
        """初始化验证器

        Args:
            file_path: 日志记录器，如果为None则使用默认配置
        """

        self.input_data = load_sharegpt_data(file_path)

    def validate_output(self):
        """验证输出结果

        需要和场景、分簇触发逻辑、后续任务规划环节结合起来.

        1、正确性验证（50分）：
            输出的目标、卫星是否在输入的数据范围内；
            输出的目标是否全部覆盖了输入的数据。

            确保分簇输出在t时刻覆盖t时刻的输入数据，并预测t+Δt的覆盖（使用Kalman滤波或粒子滤波来处理不确定性）。惩罚机制：如果覆盖率<95%，直接扣满分。
        2、分簇稳定性（20分）:
            目标，如果能被上一次分簇观测到，但是这次不再属于这个簇，惩罚
            卫星，如果在当前的簇还能正常工作，但是被划分到了其他的簇，惩罚

            引入Hysteresis机制（滞回阈值）：只有当收益>阈值时才允许切换簇。量化惩罚：用Jaccard相似度衡量前后簇的重叠率，低于80%扣分。
        3、通信代价: （10分）
            簇内同步代价：1x distance
            全网同步代价：1.2x master node distance，涉及到主节点的选择。
        4、观测效能评估（10分）：
            同一个簇内，目标被两颗卫星同时观测的概率
        5、簇间隔离性（5分）：
            一个目标，仅可以被一个簇观测
            一个卫星，只能属于一个簇
        6、分簇规模（5分）：
            小于等于2，大于等于10，都不合适。

        Args:

        Returns:
            验证结果，包含验证状态、错误信息和警告信息
        """
        validation_result = [
            ValidationItem(
                input=conversation.input,
                response=conversation.response.clusters,
                validation_details=[],
            )
            for conversation in self.input_data
        ]

        try:
            # 1. 目标覆盖验证，含：不存在的目标，目标覆盖情况
            self._correctness_validation(self.input_data, validation_result)

        except Exception as e:
            print(f"验证过程发生异常: {e}")

        return validation_result

    def _correctness_validation(
            self,
            input_data: List[LLMConversationMessage],
            result: List[ValidationItem],
    ) -> None:
        """验证卫星分配

        Args:
            input_data:
            result:
        """

        for index, conversation in tqdm(enumerate(input_data)):
            # 提取输入中的所有目标
            input_targets = set()
            for edge in conversation.input.target_edges:
                input_targets.add(edge.target_id)

            # 提取输出中的所有目标
            output_targets = set()
            for cluster in conversation.response.clusters:
                output_targets.update(cluster.targets)
            # 计算覆盖率
            if input_targets:
                coverage_rate = round(len(output_targets & input_targets) / len(input_targets), 2)
            else:
                coverage_rate = 1.0  # 如果没有输入目标，认为覆盖率为100%

            # 创建验证详情
            from data_class.data_validation_models import ValidationDetail
            result[index].validation_details.append(ValidationDetail(
                validation_type="correctness_validation",
                score=0,
                info=f"覆盖率: {coverage_rate:.1%}"
            ))


def plot_coverage(results: List[ValidationItem], save_path: Optional[str] = None, image_title: str = "数据推理结果"):
    """
    绘制输入目标数量 vs 目标覆盖率的气泡图，气泡大小表示数据点数量，
    并在图片下方添加覆盖率分段统计注释

    参数:
    results -- ValidationItem 对象列表
    save_path -- 可选，图表保存路径
    """
    # 提取数据
    input_nums = []
    coverage_rates = []

    for res in results:
        # 从validation_details中找到correctness_validation数据
        coverage_detail = None
        for detail in res.validation_details:
            if detail.validation_type == "correctness_validation":
                coverage_detail = detail
                # break

        if coverage_detail:
            # 从info中解析覆盖率信息
            if "覆盖率:" in coverage_detail.info:
                try:
                    # 解析覆盖率，例如 "覆盖率: 85.0%"
                    coverage_str = coverage_detail.info.split("覆盖率:")[1].strip().rstrip('%')
                    coverage_rate = float(coverage_str) / 100

                    # 估算输入目标数量（这里简化处理）
                    input_num = len(res.input.target_edges) if hasattr(res.input, 'target_edges') else 10

                    input_nums.append(input_num)
                    coverage_rates.append(coverage_rate)
                except (ValueError, IndexError):
                    continue

    if not input_nums:
        print("没有有效数据可绘制")
        return

    # 统计覆盖率分段数量
    total = len(coverage_rates)
    count_gt_100 = sum(1 for r in coverage_rates if r > 1.0)
    count_100 = sum(1 for r in coverage_rates if r == 1.0)
    count_90_100 = sum(1 for r in coverage_rates if 0.9 <= r < 1.0)
    count_80_90 = sum(1 for r in coverage_rates if 0.8 <= r < 0.9)
    count_lt_80 = sum(1 for r in coverage_rates if r < 0.8)

    # 构建注释文本
    annotation = (
        f"覆盖率分段统计（样本总数: {total})：\n"
        f">100%: {count_gt_100} ({count_gt_100 / total:.1%})    "
        f"100%: {count_100} ({count_100 / total:.1%})    "
        f"90~100%: {count_90_100} ({count_90_100 / total:.1%})    "
        f"80~90%: {count_80_90} ({count_80_90 / total:.1%})    "
        f"<80%: {count_lt_80} ({count_lt_80 / total:.1%})"
    )

    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 8))

    # 计算点的频率用于气泡大小
    data_points = list(zip(input_nums, coverage_rates))
    point_counter = Counter(data_points)

    # 绘制气泡图
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=min(coverage_rates), vmax=max(coverage_rates))

    # 创建一个ScalarMappable用于颜色映射
    cmap = plt.colormaps['viridis']
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # 必须设置一个空数组

    # 绘制气泡图
    for (x, y), count in point_counter.items():
        ax.scatter(x, y, s=count * 50, alpha=0.6,
                   color=cmap(norm(y)), edgecolors='black', linewidth=0.5)

    # 尝试使用系统中已安装的中文字体
    try:
        font_path = get_project_root() / "utils/simhei.ttf"
        chinese_font = FontProperties(fname=str(font_path))
    except Exception:
        print("警告：无法加载中文字体，将使用系统默认字体")
        chinese_font = FontProperties()

    # 添加标签和标题
    ax.set_xlabel("输入目标数量", fontproperties=chinese_font, fontsize=12, labelpad=10)
    ax.set_ylabel("结果目标覆盖率", fontproperties=chinese_font, fontsize=12, labelpad=10)
    ax.set_title(
        image_title,
        fontproperties=chinese_font, fontsize=14, pad=20
    )

    # 设置坐标轴范围
    ax.set_xlim(min(input_nums) - 1, max(input_nums) + 1)
    ax.set_ylim(min(coverage_rates) - 0.05, max(coverage_rates) + 0.02)

    # 添加网格和样式优化
    ax.grid(True, linestyle="--", alpha=0.6)

    # 添加颜色条（使用先前创建的ScalarMappable）
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.ax.set_ylabel("结果目标覆盖率", fontproperties=chinese_font)

    # 在图片下方添加注释
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.15)  # 留出空间放注释
    fig.text(
        0.48, 0.02, annotation, ha="center", va="bottom",
        fontsize=12, fontproperties=chinese_font,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.2')
    )

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"图表已保存至: {save_path}")
    else:
        plt.tight_layout()
        plt.show()

    plt.close()


def main():
    timestamp = get_current_timestamp()

    data_path = (
            get_data_dir() / "cluster_results_sharegpt_training_data/clustering_results_cmax_2000111.jsonl"
    )
    out_path = get_data_dir() / f"cluster_results_sharegpt_training_data/{data_path.stem}_coverage_{timestamp}.png"
    # data_path = (
    #     get_data_dir() / "training_data_sharegpt_qwen3_235B_A22B_20250626_113625_30_v2.jsonl"
    # )
    # data_path = (
    #     get_data_dir() / "training_data_sharegpt_gemini-2.5-pro_20250629_103625_30_v3.jsonl"
    # )
    # data_path = (
    #     get_data_dir() / "training_data_sharegpt_qwen3-4b_20250701_191138_24_v5.jsonl"
    # )
    # image_title = "Qwen3 235B-A22B 数据推理结果"
    # image_title = "Google Gemini-2.5-pro-250605 数据推理结果"
    # image_title = "OpenAI GPT-o3/o4mini 数据推理结果"
    # image_title = "Qwen3 4B 数据推理结果"
    image_title = "max_overlap_alg"
    validator = ClusterDataValidator(file_path=data_path)
    data = validator.validate_output()

    plot_coverage(data, save_path=str(out_path),
                  image_title=image_title)
    print(len(data))


if __name__ == "__main__":
    main()
    print("Done.")
