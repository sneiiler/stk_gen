"""
数据验证器模块

该模块提供了用于验证大模型生成的卫星分簇结果的验证器类。
"""

import sys
from typing import Dict, Any, List, Set, Tuple, Optional
from collections import Counter  # 在文件顶部导入

sys.path.append("/root/stk_gen")
import json, re
import logging
from datetime import datetime
from pydantic import BaseModel
from collections import defaultdict
from icecream import ic, install
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

install()
from utils.misc_utils import get_data_dir, get_project_root


class ValidationInput(BaseModel):
    input_user_data: List[dict]
    output_resoning_data: List[str]
    output_result_data: List[list]


class ValidationResult(BaseModel):
    data_id: str
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    details: Dict[str, Any]


class ClusterDataValidator:
    """卫星分簇结果验证器

    用于验证大模型生成的卫星分簇结果是否符合业务规则和约束条件。
    """

    def __init__(self, file_path):
        """初始化验证器

        Args:
            file_path: 日志记录器，如果为None则使用默认配置
        """

        self.input_data = self.load_data(file_path)

    def load_data(self, file_path: str) -> ValidationInput:
        """加载数据

        Args:
            file_path: 数据文件路径

        Returns:
            数据列表
        """
        # ensure file_path is string for suffix checks
        file_path_str = str(file_path)
        raw_data = []
        input_user_data = []
        output_resoning_data = []
        output_result_data = []

        if file_path_str.endswith(".json"):
            # 如果是JSON文件，直接加载
            with open(file_path, "r", encoding="utf-8") as file:
                raw_data = json.load(file)
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
        
        for line_index, line_data in enumerate(raw_data):
            for message in line_data["messages"]:
                if message["role"] == "user":
                    input_user_data.append(json.loads(message["content"]))
                if message["role"] == "assistant":
                    # 使用正则表达式匹配 </thionk> 后面到下一个 ``` 之间的内容
                    pattern = r"<think>(.*?)</think>(.*?)\[(.*)\]$$"
                    match = re.search(pattern, message["content"], re.DOTALL)
                    if match:
                        try:
                            output_resoning_data.append(match.group(1))
                            # 先去除所有反斜杠，避免解析失败
                            group3_cleaned = match.group(3).replace("\\", "")
                            output_result_data.append(
                                json.loads("[" + group3_cleaned + "]")
                            )
                        except json.JSONDecodeError as e:
                            print("JSON 解析失败:", e)
                            raise ValueError(f"JSON 解析失败: {e}")
        return ValidationInput(
            input_user_data=input_user_data,
            output_resoning_data=output_resoning_data,
            output_result_data=output_result_data,
        )

    def validate_output(self):
        """验证输出结果

        Args:

        Returns:
            验证结果，包含验证状态、错误信息和警告信息
        """
        validation_result = [
            ValidationResult(
                data_id=item["timestamp"],
                is_valid=True,
                errors=[],
                warnings=[],
                details={},
            )
            for item in self.input_data.input_user_data
        ]

        try:
            # 1. 目标覆盖验证，含：不存在的目标，目标覆盖情况
            self._validate_target_coverage(self.input_data, validation_result)

            # 2. 验证卫星分配情况，含：同一卫星分配到多个簇，不存在的卫星，卫星使用率
            self._validate_satellite_assignment(self.input_data, validation_result)

            # 3. 最小分簇验证
            self._validate_minimize_clusters(self.input_data, validation_result)

            # # 5. 分簇验证
            # self._validate_cluster_qulitys(self.input_data, validation_result)

            # # 6. 链路质量验证
            # self._validate_link_quality(output, self., validation_result)

            # # 7. 观测质量验证
            # self._validate_observation_quality(output, input_data, validation_result)

        except Exception as e:
            print(f"验证过程发生异常: {e}")

        return validation_result

    def _validate_target_coverage(
        self,
        input_data: ValidationInput,
        result: List[ValidationResult],
    ) -> None:
        """验证卫星分配

        Args:
        """
        input_user_data = input_data.input_user_data
        output_result_data = input_data.output_result_data

        for index, row in tqdm(enumerate(input_user_data)):
            # 提取输入中的所有目标
            input_targets = set()
            for edge in row.get("target_edges", []):
                input_targets.add(edge["to"])

            # 提取输出中的所有目标
            output_targets = set()
            for cluster in output_result_data[index]:
                output_targets.update(cluster.get("targets", []))

            # 检查目标覆盖是否完整
            missing_targets = input_targets - output_targets
            extra_targets = output_targets - input_targets

            if missing_targets:
                result[index].is_valid = False
                result[index].errors.append(
                    f"目标覆盖不完整：缺少目标 {missing_targets}"
                )

            if extra_targets:
                result[index].is_valid = False
                result[index].errors.append(
                    f"目标覆盖错误：包含不存在的目标 {extra_targets}"
                )

            result[index].details["target_coverage"] = {
                "input_targets": input_targets,
                "output_targets": output_targets,
                "input_targets_num": len(input_targets),
                "output_targets_num": len(output_targets),
                "coverage_rate": (
                    round(len(output_targets & input_targets) / len(input_targets), 2)
                    if input_targets
                    else 0
                ),
            }

    def _validate_satellite_assignment(
        self,
        input_data: ValidationInput,
        result: List[ValidationResult],
    ) -> None:
        """验证卫星分配

        Args:
            output: 输出数据
            input_data: 输入数据
            result: 验证结果对象
        """
        input_user_data = input_data.input_user_data
        output_result_data = input_data.output_result_data

        for index, row in tqdm(enumerate(input_user_data)):
            # 提取输入中的所有卫星
            input_satellites = set()
            for attr in row.get("sat_attrs", []):
                input_satellites.add(attr["id"])

            satellite_assignments = defaultdict(list)
            all_output_sats = []

            for cluster_idx, cluster in enumerate(output_result_data[index]):
                cluster_sats = cluster.get("sats", [])
                all_output_sats.extend(cluster_sats)

                for sat in cluster_sats:
                    satellite_assignments[sat].append(cluster_idx)

            output_satellites = set(all_output_sats)

            # 检查重复分配
            duplicate_assignments = {
                sat: clusters
                for sat, clusters in satellite_assignments.items()
                if len(clusters) > 1
            }
            if duplicate_assignments:
                result[index].is_valid = False
                for sat, clusters in duplicate_assignments.items():
                    result[index].warnings.append(
                        f"卫星 {sat} 被分配到多个簇: {clusters}"
                    )

            # 检查是否存在不存在的卫星
            invalid_satellites = output_satellites - input_satellites

            if invalid_satellites:
                result[index].is_valid = False
                result[index].errors.append(
                    f"包含不存在的卫星: {sorted(invalid_satellites)}"
                )

            # 检查卫星利用率
            unused_satellites = input_satellites - output_satellites
            if unused_satellites:
                result[index].warnings.append(
                    f"未使用的卫星: {sorted(unused_satellites)}"
                )

            result[index].details["satellite_assignment"] = {
                "total_satellites": len(input_satellites),
                "assigned_satellites": len(output_satellites),
                "utilization_rate": (
                    len(output_satellites) / len(input_satellites)
                    if input_satellites
                    else 0
                ),
                "unused_satellites": sorted(unused_satellites),
                "duplicate_assignments": duplicate_assignments,
            }

    def _validate_minimize_clusters(
        self,
        input_data: ValidationInput,
        result: List[ValidationResult],
    ) -> None:
        """验证最小分簇原则
        Args:
        """
        input_user_data = input_data.input_user_data
        output_result_data = input_data.output_result_data

        for index, row in tqdm(enumerate(output_result_data)):
            # 检查是否有可合并的簇
            cluster_count = len(row)

            # 仅当簇数量较多时检查合并可能性
            if cluster_count > 1:
                # 构建簇间连接强度映射
                sat_cluster_map = {}
                for cluster_idx, cluster in enumerate(row):
                    for sat in cluster.get("sats", []):
                        sat_cluster_map[sat] = cluster_idx

                # 计算簇间连接强度
                inter_cluster_strength = defaultdict(float)
                for edge in input_user_data[index].get("sat_edges", []):
                    from_sat, to_sat = edge["from"], edge["to"]
                    from_cluster = sat_cluster_map.get(from_sat)
                    to_cluster = sat_cluster_map.get(to_sat)

                    # 检查卫星是否被分配
                    if from_cluster is not None and to_cluster is not None:
                        # 属于不同簇
                        if from_cluster != to_cluster:
                            # 使用簇对作为键，确保顺序一致
                            cluster_pair = tuple(sorted((from_cluster, to_cluster)))
                            inter_cluster_strength[cluster_pair] += edge["w"]

                # 检查是否存在强连接（强度 > 0.7）
                merge_candidates = []
                for (c1, c2), strength in inter_cluster_strength.items():
                    if strength > 0.7:
                        merge_candidates.append((c1, c2, strength))

                if merge_candidates:
                    result[index].warnings.append(
                        f"发现 {len(merge_candidates)} 对可合并簇 (簇间连接强度 > 0.7)"
                    )
                    result[index].details["merge_candidates"] = merge_candidates


def plot_coverage(results: List[ValidationResult], save_path: Optional[str] = None):
    """
    绘制输入目标数量 vs 目标覆盖率的气泡图，气泡大小表示数据点数量，
    并在图片下方添加覆盖率分段统计注释

    参数:
    results -- ValidationResult 对象列表
    save_path -- 可选，图表保存路径
    """
    # 提取数据
    input_nums = []
    coverage_rates = []

    for res in results:
        # 直接从details字典中获取数据
        coverage_data = res.details.get("target_coverage", {})
        input_num = coverage_data.get("input_targets_num")
        coverage_rate = coverage_data.get("coverage_rate")

        # 确保数据有效
        if input_num is not None and coverage_rate is not None:
            input_nums.append(input_num)
            coverage_rates.append(coverage_rate)

    if not input_nums:
        print("没有有效数据可绘制")
        return

    # 统计覆盖率分段数量
    total = len(coverage_rates)
    count_100 = sum(1 for r in coverage_rates if r == 1.0)
    count_90_100 = sum(1 for r in coverage_rates if 0.9 <= r < 1.0)
    count_80_90 = sum(1 for r in coverage_rates if 0.8 <= r < 0.9)
    count_lt_80 = sum(1 for r in coverage_rates if r < 0.8)

    # 构建注释文本
    annotation = (
        f"覆盖率分段统计：\n"
        f"100%: {count_100} ({count_100/total:.1%})    "
        f"90~100%: {count_90_100} ({count_90_100/total:.1%})    "
        f"80~90%: {count_80_90} ({count_80_90/total:.1%})    "
        f"<80%: {count_lt_80} ({count_lt_80/total:.1%})"
    )

    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 计算点的频率用于气泡大小
    data_points = list(zip(input_nums, coverage_rates))
    point_counter = Counter(data_points)
    
    # 绘制气泡图
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    norm = Normalize(vmin=min(coverage_rates), vmax=max(coverage_rates))
    
    # 创建一个ScalarMappable用于颜色映射
    cmap = plt.colormaps['viridis']  # 使用推荐的新方法
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # 必须设置一个空数组
    
    # 绘制气泡图
    for (x, y), count in point_counter.items():
        ax.scatter(x, y, s=count*50, alpha=0.6, 
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
    ax.set_ylabel("目标覆盖率", fontproperties=chinese_font, fontsize=12, labelpad=10)
    ax.set_title(
        "输入目标数量 vs 目标覆盖率 v1(气泡大小表示数据点数量)", 
        fontproperties=chinese_font, fontsize=14, pad=20
    )

    # 设置坐标轴范围
    ax.set_xlim(min(input_nums) - 1, max(input_nums) + 1)
    ax.set_ylim(min(coverage_rates) - 0.05, 1.05)

    # 添加网格和样式优化
    ax.grid(True, linestyle="--", alpha=0.6)
    
    # 添加颜色条（使用先前创建的ScalarMappable）
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.ax.set_ylabel("覆盖率", fontproperties=chinese_font)

    # 在图片下方添加注释
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.15)  # 留出空间放注释
    fig.text(
        0.5, 0.02, annotation, ha="center", va="bottom", 
        fontsize=12, fontproperties=chinese_font,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5')
    )

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图表已保存至: {save_path}")
    else:
        plt.tight_layout()
        plt.show()
    
    plt.close()


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    data_path = (
        get_data_dir() / "distilled_training_data_v20250618_sharegpt_format_v1.json"
    )
    # data_path = (
    #     get_data_dir() / "distilled_training_data_v20250626_sharegpt_format_v2.jsonl"
    # )
    data_path = (
        get_data_dir() / "training_data_sharegpt_gemini-2.5-pro_20250629_103625_30_v3.jsonl"
    )
    validator = ClusterDataValidator(file_path=data_path)
    data = validator.validate_output()

    plot_coverage(data, save_path=str(get_data_dir() / f"coverage_{timestamp}_{str(data_path)[-44:-4]}.png"))
    print(len(data))


if __name__ == "__main__":
    main()
    print("Done.")
