"""
数据验证器模块

该模块提供了用于验证大模型生成的卫星分簇结果的验证器类。
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到路径
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from collections import defaultdict
from icecream import ic
from tqdm import tqdm
from utils.misc_utils import get_current_timestamp, get_data_dir
from misc_tools.sharegpt_utils import load_sharegpt_data
from data_classes.sft_data_models import LLMConversationMessage
from data_classes.data_validation_models import ValidationItem, ValidationDetail


class ClusterDataValidator:
    """卫星分簇结果验证器

    用于验证大模型生成的卫星分簇结果是否符合业务规则和约束条件。
    """

    def __init__(self):
        """初始化验证器

        Args:
        """

    def validate_output(
        self, input_data: List[LLMConversationMessage]
    ) -> List[ValidationItem]:
        """验证输出结果

        验证体系（每项100分，加权求和得到最终得分）：
        
        1、正确性验证（100分，权重40%）：
            输出的目标、卫星是否在输入的数据范围内；
            输出的目标是否全部覆盖了输入的数据。
            一个目标，仅可以被一个簇观测
            一个卫星，只能属于一个簇

            确保分簇输出在t时刻覆盖t时刻的输入数据，惩罚机制：如果覆盖率<95%，直接扣满分。

        2、分簇稳定性（100分，权重30%）:
            目标，如果能被上一次分簇观测到，但是这次不再属于这个簇，惩罚
            卫星，如果在当前的簇还能正常工作，但是被划分到了其他的簇，惩罚

            引入Hysteresis机制（滞回阈值）：只有当收益>阈值时才允许切换簇。量化惩罚：用Jaccard相似度衡量前后簇的重叠率，低于80%扣分。

        3、通信代价（100分，权重10%）:
            簇内同步代价：1x distance
            全网同步代价：1.2x master node distance，涉及到主节点的选择。

        4、观测效能评估（100分，权重10%）：
            同一个簇内，目标被两颗卫星同时观测的概率
            
        5、分簇规模（100分，权重10%）：
            小于等于2，大于等于10，都不合适。
            
        最终得分 = 正确性×0.4 + 稳定性×0.3 + 通信代价×0.1 + 观测效能×0.1 + 分簇规模×0.1

        Args:
            input_data: 输入数据列表

        Returns:
            验证结果，包含验证状态、错误信息和警告信息
        """
        validation_results = []

        try:
            # 按时间切片进行验证，显示总体进度
            for conversation in tqdm(input_data, desc="验证时间切片"):
                # 为每个时间切片创建验证结果项
                validation_details = []
                
                # 1. 正确性和隔离性验证（100分，权重40%）
                correctness_detail = self._validate_correctness_and_isolation_for_single_slice(conversation)
                validation_details.append(correctness_detail)

                # 2. 分簇稳定性验证（100分，权重30%）
                stability_detail = self._validate_stability_for_single_slice(conversation)
                validation_details.append(stability_detail)

                # 3. 通信代价验证（100分，权重10%）
                cost_detail = self._validate_communication_cost_for_single_slice(conversation)
                validation_details.append(cost_detail)

                # 4. 观测效能验证（100分，权重10%）
                efficiency_detail = self._validate_observation_efficiency_for_single_slice(conversation)
                validation_details.append(efficiency_detail)

                # 5. 分簇规模验证（100分，权重10%）
                size_detail = self._validate_cluster_size_for_single_slice(conversation)
                validation_details.append(size_detail)
                
                # 创建单个时间切片的验证结果
                validation_item = ValidationItem(
                    input=conversation.input,
                    response=conversation.response.clusters,
                    validation_details=validation_details,
                )
                validation_results.append(validation_item)

        except Exception as e:
            print(f"验证过程发生异常: {e}")

        return validation_results

    def _validate_correctness_and_isolation_for_single_slice(
        self,
        conversation: LLMConversationMessage,
    ) -> ValidationDetail:
        """正确性和隔离性验证（100分）- 单个时间切片版本
        
        验证内容：
        1. 致命错误检测：分簇中出现不存在于输入的卫星-目标连接 → 扣全部100分，标记ERROR错误
        2. 目标遗漏检测：输入中的目标未被覆盖 → 按比例扣分（最多50分）
        3. 隔离性验证：
           3.1 卫星-目标连接跨簇：卫星和目标出现在不同簇中 → 按比例扣分（最多25分）
           3.2 多簇归属：同一个卫星或目标出现在多个分簇中 → 按数量扣分（最多25分）

        Args:
            conversation: 单个对话数据
            
        Returns:
            ValidationDetail: 验证详情对象
        """
        # 构建输入中的有效卫星-目标连接集合
        valid_sat_target_connections = set()
        input_targets = set()
        input_satellites = set()
        
        for edge in conversation.input.target_edges:
            valid_sat_target_connections.add((edge.sat_id, edge.target_id))
            input_targets.add(edge.target_id)
            input_satellites.add(edge.sat_id)
        
        # 构建输出分簇的信息
        output_targets = set()
        output_satellites = set()
        target_cluster_map = {}  # 目标 -> 簇ID
        satellite_cluster_map = {}  # 卫星 -> 簇ID
        
        for cluster_idx, cluster in enumerate(conversation.response.clusters):
            cluster_targets = set(cluster.targets)
            cluster_satellites = set(cluster.sats)
            
            # 记录目标和卫星的簇归属
            for target in cluster_targets:
                target_cluster_map[target] = cluster_idx
                output_targets.add(target)
            
            for satellite in cluster_satellites:
                satellite_cluster_map[satellite] = cluster_idx
                output_satellites.add(satellite)
        
        # === 1. 致命错误检测：输出中包含不存在的目标或卫星 ===
        invalid_targets = output_targets - input_targets  # 输出中存在但输入中不存在的目标
        invalid_satellites = output_satellites - input_satellites  # 输出中存在但输入中不存在的卫星
        
        if invalid_targets or invalid_satellites:
            # 发现致命错误，直接扣满分并标记critic错误
            error_details = []
            
            if invalid_targets:
                target_list = sorted(list(invalid_targets))[:10]  # 只显示前10个
                if len(invalid_targets) > 10:
                    target_list.append(f"...还有{len(invalid_targets)-10}个")
                error_details.append(f"不存在的目标: {', '.join(target_list)}")
            
            if invalid_satellites:
                sat_list = sorted(list(invalid_satellites))[:10]  # 只显示前10个
                if len(invalid_satellites) > 10:
                    sat_list.append(f"...还有{len(invalid_satellites)-10}个")
                error_details.append(f"不存在的卫星: {', '.join(sat_list)}")
            
            error_msg = f"[ERROR] 分簇中包含输入中不存在的元素: {'; '.join(error_details)}"
            
            return ValidationDetail(
                validation_type="correctness_validation",
                score=0,  # 直接扣满分
                info=error_msg,
            )
        
        # === 2. 目标遗漏检测 ===
        missing_targets = input_targets - output_targets
        target_coverage_rate = len(output_targets & input_targets) / len(input_targets) if input_targets else 1.0
        
        # === 3. 隔离性验证 ===
        
        # 3.1 检查卫星-目标连接的跨簇情况
        cross_cluster_violations = []
        total_valid_connections = 0
        
        for sat, target in valid_sat_target_connections:
            # 检查该连接在输出中是否存在且跨簇
            sat_cluster = satellite_cluster_map.get(sat)
            target_cluster = target_cluster_map.get(target)
            
            # 只考虑在输出中都存在的卫星和目标
            if sat_cluster is not None and target_cluster is not None:
                total_valid_connections += 1
                if sat_cluster != target_cluster:
                    cross_cluster_violations.append({
                        'satellite': sat,
                        'target': target,
                        'sat_cluster': sat_cluster,
                        'target_cluster': target_cluster
                    })
        
        # 3.2 检查同一个卫星/目标是否出现在多个分簇中
        satellite_multi_cluster_violations = []
        target_multi_cluster_violations = []
        
        # 检查每个卫星是否只出现在一个簇中
        satellite_to_clusters = defaultdict(list)
        target_to_clusters = defaultdict(list)
        
        for cluster_idx, cluster in enumerate(conversation.response.clusters):
            for satellite in cluster.sats:
                satellite_to_clusters[satellite].append(cluster_idx)
            for target in cluster.targets:
                target_to_clusters[target].append(cluster_idx)
        
        # 找出出现在多个簇中的卫星
        for satellite, clusters in satellite_to_clusters.items():
            if len(clusters) > 1:
                satellite_multi_cluster_violations.append({
                    'satellite': satellite,
                    'clusters': clusters
                })
        
        # 找出出现在多个簇中的目标
        for target, clusters in target_to_clusters.items():
            if len(clusters) > 1:
                target_multi_cluster_violations.append({
                    'target': target,
                    'clusters': clusters
                })
        
        # === 计算扣分 ===
        score_penalty = 0
        errors = []
        warnings = []
        
        # 目标遗漏扣分（最多50分）
        if missing_targets:
            coverage_penalty = (1.0 - target_coverage_rate) * 50  # 覆盖率不足扣分
            score_penalty += coverage_penalty
            errors.append(f"[ERROR] 目标遗漏：缺少{len(missing_targets)}个目标({target_coverage_rate:.1%}覆盖率)，扣{coverage_penalty:.1f}分")
            
            # 显示具体遗漏的目标（前5个）
            missing_list = sorted(list(missing_targets))[:5]
            if len(missing_targets) > 5:
                missing_list.append(f"...还有{len(missing_targets)-5}个")
            warnings.append(f"[WARNING] 遗漏目标详情: {', '.join(missing_list)}")
        
        # 隔离性违反扣分（最多50分）
        isolation_penalty = 0
        
        # 3.1 卫星-目标连接跨簇扣分（最多25分）
        if cross_cluster_violations and total_valid_connections > 0:
            cross_cluster_rate = len(cross_cluster_violations) / total_valid_connections
            connection_isolation_penalty = cross_cluster_rate * 25  # 按比例扣分，最多25分
            isolation_penalty += connection_isolation_penalty
            
            errors.append(f"[ERROR] 连接跨簇违反：{len(cross_cluster_violations)}/{total_valid_connections}({cross_cluster_rate:.1%})的有效连接跨簇，扣{connection_isolation_penalty:.1f}分")
            
            # 显示具体的跨簇连接（前5个）
            violation_details = []
            for violation in cross_cluster_violations[:5]:
                violation_details.append(f"卫星{violation['satellite']}(簇{violation['sat_cluster']})-目标{violation['target']}(簇{violation['target_cluster']})")
            
            if len(cross_cluster_violations) > 5:
                violation_details.append(f"...还有{len(cross_cluster_violations)-5}个跨簇连接")
            
            warnings.append(f"[WARNING] 跨簇连接详情: {'; '.join(violation_details)}")
        
        # 3.2 卫星/目标多簇出现扣分（最多25分）
        multi_cluster_penalty = 0
        
        if satellite_multi_cluster_violations:
            # 每个多簇卫星扣5分，最多12.5分
            satellite_penalty = min(len(satellite_multi_cluster_violations) * 5, 12.5)
            multi_cluster_penalty += satellite_penalty
            
            sat_details = []
            for violation in satellite_multi_cluster_violations[:3]:
                sat_details.append(f"卫星{violation['satellite']}出现在簇{violation['clusters']}")
            if len(satellite_multi_cluster_violations) > 3:
                sat_details.append(f"...还有{len(satellite_multi_cluster_violations)-3}个")
            
            errors.append(f"[ERROR] 卫星多簇违反：{len(satellite_multi_cluster_violations)}个卫星出现在多个簇中，扣{satellite_penalty:.1f}分")
            warnings.append(f"[WARNING] 多簇卫星详情: {'; '.join(sat_details)}")
        
        if target_multi_cluster_violations:
            # 每个多簇目标扣5分，最多12.5分
            target_penalty = min(len(target_multi_cluster_violations) * 5, 12.5)
            multi_cluster_penalty += target_penalty
            
            target_details = []
            for violation in target_multi_cluster_violations[:3]:
                target_details.append(f"目标{violation['target']}出现在簇{violation['clusters']}")
            if len(target_multi_cluster_violations) > 3:
                target_details.append(f"...还有{len(target_multi_cluster_violations)-3}个")
            
            errors.append(f"[ERROR] 目标多簇违反：{len(target_multi_cluster_violations)}个目标出现在多个簇中，扣{target_penalty:.1f}分")
            warnings.append(f"[WARNING] 多簇目标详情: {'; '.join(target_details)}")
        
        isolation_penalty += multi_cluster_penalty
        score_penalty += isolation_penalty
        
        # 确保扣分不超过100分
        score_penalty = min(score_penalty, 100)
        final_score = int(100 - score_penalty)
        
        # 构建信息文本
        cross_cluster_rate = len(cross_cluster_violations) / total_valid_connections if total_valid_connections > 0 else 0
        multi_sat_count = len(satellite_multi_cluster_violations)
        multi_target_count = len(target_multi_cluster_violations)
        
        info_text = (f"目标覆盖率: {target_coverage_rate:.1%}({len(output_targets & input_targets)}/{len(input_targets)}), "
                    f"连接跨簇率: {cross_cluster_rate:.1%}({len(cross_cluster_violations)}/{total_valid_connections}), "
                    f"多簇卫星: {multi_sat_count}个, 多簇目标: {multi_target_count}个, "
                    f"输入目标数: {len(input_targets)}, 输出目标数: {len(output_targets)}, "
                    f"输入卫星数: {len(input_satellites)}, 输出卫星数: {len(output_satellites)}, "
                    f"得分: {final_score}/100")
        if errors:
            info_text += f" | 错误: {'; '.join(errors)}"
        if warnings:
            info_text += f" | 警告: {'; '.join(warnings)}"

        return ValidationDetail(
            validation_type="correctness_validation",
            score=final_score,
            info=info_text,
        )


    def _validate_stability_for_single_slice(
        self,
        conversation: LLMConversationMessage,
    ) -> ValidationDetail:
        """分簇稳定性验证（100分）- 单个时间切片版本

        验证内容：
        1. 目标，如果能被上一次分簇观测到，但是这次不再属于这个簇，惩罚
        2. 卫星，如果在当前的簇还能正常工作，但是被划分到了其他的簇，惩罚
        3. 引入Hysteresis机制（滞回阈值）：只有当收益>阈值时才允许切换簇
        4. 量化惩罚：用Jaccard相似度衡量前后簇的重叠率，低于80%扣分

        Args:
            conversation: 单个对话数据

        Returns:
            ValidationDetail: 验证详情对象
        """
        errors = []
        warnings = []
        score_penalty = 0

        # 检查是否有历史分簇结果
        history_results = conversation.input.history_cluster_result
        if not history_results or len(history_results) == 0:
            # 没有历史数据，无法进行稳定性验证，直接给满分
            return ValidationDetail(
                validation_type="stability_validation",
                score=100,
                info="[WARNING] 无历史分簇数据，跳过稳定性验证",
            )

        # 获取最近一次的历史分簇结果
        last_clusters = history_results[-1] if history_results else []
        current_clusters = conversation.response.clusters

        if not last_clusters:
            return ValidationDetail(
                validation_type="stability_validation",
                score=0,
                info="[WARNING] 无有效历史分簇数据，跳过稳定性验证",
            )

        # 构建历史分簇的目标和卫星映射
        last_target_to_cluster = {}
        last_satellite_to_cluster = {}

        for cluster_idx, cluster in enumerate(last_clusters):
            for target in cluster.targets:
                last_target_to_cluster[target] = cluster_idx
            for satellite in cluster.sats:
                last_satellite_to_cluster[satellite] = cluster_idx

        # 构建当前分簇的目标和卫星映射
        current_target_to_cluster = {}
        current_satellite_to_cluster = {}

        for cluster_idx, cluster in enumerate(current_clusters):
            for target in cluster.targets:
                current_target_to_cluster[target] = cluster_idx
            for satellite in cluster.sats:
                current_satellite_to_cluster[satellite] = cluster_idx

        # 1. 检查目标稳定性
        target_switches = 0
        total_targets = len(set(last_target_to_cluster.keys()) | set(current_target_to_cluster.keys()))

        for target in last_target_to_cluster:
            if target in current_target_to_cluster:
                if last_target_to_cluster[target] != current_target_to_cluster[target]:
                    target_switches += 1
                    warnings.append(f"[WARNING] 目标 {target} 从簇 {last_target_to_cluster[target]} 切换到簇 {current_target_to_cluster[target]}")

        # 2. 检查卫星稳定性
        satellite_switches = 0
        total_satellites = len(set(last_satellite_to_cluster.keys()) | set(current_satellite_to_cluster.keys()))

        for satellite in last_satellite_to_cluster:
            if satellite in current_satellite_to_cluster:
                if last_satellite_to_cluster[satellite] != current_satellite_to_cluster[satellite]:
                    satellite_switches += 1
                    warnings.append(f"[WARNING] 卫星 {satellite} 从簇 {last_satellite_to_cluster[satellite]} 切换到簇 {current_satellite_to_cluster[satellite]}")

        # 3. 计算Jaccard相似度
        jaccard_similarities = []
        max_clusters = max(len(last_clusters), len(current_clusters))

        for i in range(max_clusters):
            last_cluster_items = set()
            current_cluster_items = set()

            if i < len(last_clusters):
                last_cluster_items = set(last_clusters[i].targets + last_clusters[i].sats)
            if i < len(current_clusters):
                current_cluster_items = set(current_clusters[i].targets + current_clusters[i].sats)

            if last_cluster_items or current_cluster_items:
                intersection = len(last_cluster_items & current_cluster_items)
                union = len(last_cluster_items | current_cluster_items)
                jaccard = intersection / union if union > 0 else 0
                jaccard_similarities.append(jaccard)

        avg_jaccard = sum(jaccard_similarities) / len(jaccard_similarities) if jaccard_similarities else 0

        # 计算分数惩罚（总共100分）
        # 目标切换惩罚（最多50分）
        if total_targets > 0:
            target_switch_rate = target_switches / total_targets
            if target_switch_rate > 0.2:  # 超过20%的目标切换
                target_penalty = min(target_switch_rate * 100, 50)
                score_penalty += target_penalty
                errors.append(f"[ERROR] 目标切换率过高: {target_switch_rate:.1%}")

        # 卫星切换惩罚（最多33分）
        if total_satellites > 0:
            satellite_switch_rate = satellite_switches / total_satellites
            if satellite_switch_rate > 0.15:  # 超过15%的卫星切换
                satellite_penalty = min(satellite_switch_rate * 67, 33)
                score_penalty += satellite_penalty
                errors.append(f"[ERROR] 卫星切换率过高: {satellite_switch_rate:.1%}")

        # Jaccard相似度惩罚（最多17分）
        if avg_jaccard < 0.8:
            jaccard_penalty = (0.8 - avg_jaccard) * 83  # 最多17分
            score_penalty += jaccard_penalty
            errors.append(f"[ERROR] 簇重叠率过低: {avg_jaccard:.1%} < 80%")

        # 确保扣分不超过100分
        score_penalty = min(score_penalty, 100)

        # 计算最终得分（满分100分减去扣分）
        final_score = int(100 - score_penalty)

        info_text = f"目标切换数: {target_switches}/{total_targets}, 卫星切换数: {satellite_switches}/{total_satellites}, 平均Jaccard相似度: {avg_jaccard:.1%}, 得分: {final_score}/100"
        if errors:
            info_text += f" | 错误: {'; '.join(errors)}"
        if warnings:
            info_text += f" | 警告: {'; '.join(warnings)}"

        return ValidationDetail(
            validation_type="stability_validation",
            score=final_score,
            info=info_text,
        )

    def _validate_communication_cost_for_single_slice(
        self,
        conversation: LLMConversationMessage,
    ) -> ValidationDetail:
        """通信代价验证（100分）- 单个时间切片版本

        验证内容：
        1. 簇内同步代价：1x distance，使用Dijkstra算法在簇内的卫星网络中寻找最短路径
        2. 全网同步代价：1x master node distance，涉及到主节点的选择
        
        扣分要点：
        完全扣分（100分）：
        - 主节点不在簇内卫星列表中
        - 簇内存在孤星（无法通过网络连接到主节点的卫星）
        
        轻微扣分（1-100分）：
        - 通信代价占星座总代价比例过高（按比例映射到1-100分）
        - 通信代价相对评估异常时的处理

        Args:
            conversation: 单个对话数据

        Returns:
            ValidationDetail: 验证详情对象
        """
        errors = []
        warnings = []
        score_penalty = 0

        # 构建卫星位置映射
        sat_positions = {}
        for sat_attr in conversation.input.sat_attrs:
            sat_positions[sat_attr.id] = sat_attr.pos

        # 构建卫星间距离映射
        sat_distances = {}
        for edge in conversation.input.sat_edges:
            sat_distances[(edge.from_sat, edge.to_sat)] = edge.distance
            sat_distances[(edge.to_sat, edge.from_sat)] = edge.distance  # 双向距离

        total_intra_cluster_cost = 0
        total_inter_cluster_cost = 0
        cluster_costs = []

        for cluster_idx, cluster in enumerate(conversation.response.clusters):
            cluster_sats = cluster.sats
            master_sat = cluster.master

            # 检查主节点是否在簇内
            if master_sat not in cluster_sats:
                errors.append(f"[ERROR] 簇 {cluster_idx} 的主节点 {master_sat} 不在簇内卫星列表中")
                score_penalty += 100
                continue

            # 计算簇内通信代价：每个成员卫星到主节点的路由代价
            intra_cluster_cost = 0
            isolated_satellites = []  # 孤星列表
            
            for member_sat in cluster_sats:
                if member_sat == master_sat:
                    continue  # 主节点自己不需要计算到自己的代价
                
                # 寻找成员卫星到主节点的最短路径
                path_cost = self._find_shortest_path_cost(
                    member_sat, master_sat, sat_distances, cluster_sats
                )
                
                if path_cost is not None:
                    intra_cluster_cost += path_cost
                else:
                    # 无法找到路径，标记为孤星
                    isolated_satellites.append(member_sat)

            total_intra_cluster_cost += intra_cluster_cost
            cluster_costs.append({
                'cluster_id': cluster_idx,
                'intra_cost': intra_cluster_cost,
                'master': master_sat,
                'size': len(cluster_sats),
                'isolated_satellites': isolated_satellites,
                'isolated_count': len(isolated_satellites)
            })

        # 计算全网同步代价（主节点间通信）
        masters = [cluster.master for cluster in conversation.response.clusters]
        all_satellites = [attr.id for attr in conversation.input.sat_attrs]  # 所有可用卫星
        
        for i, master1 in enumerate(masters):
            for master2 in masters[i+1:]:
                # 首先尝试寻找最短路径
                path_cost = self._find_shortest_path_cost(
                    master1, master2, sat_distances, all_satellites
                )
                
                if path_cost is not None:
                    # 找到路径，使用路径代价
                    total_inter_cluster_cost += path_cost * 1.0  # 1x距离系数
                else:
                    # 无法找到路径，使用欧几里得距离
                    if master1 in sat_positions and master2 in sat_positions:
                        pos1 = sat_positions[master1]
                        pos2 = sat_positions[master2]
                        distance = ((pos1[0] - pos2[0])**2 +
                                  (pos1[1] - pos2[1])**2 +
                                  (pos1[2] - pos2[2])**2)**0.5
                        total_inter_cluster_cost += distance * 1.0  # 1x距离系数

        total_cost = total_intra_cluster_cost + total_inter_cluster_cost

        # 评估通信代价合理性
        avg_cluster_cost = total_intra_cluster_cost / len(conversation.response.clusters) if conversation.response.clusters else 0

        # 检查簇内孤星并添加惩罚
        total_isolated_sats = sum(cluster['isolated_count'] for cluster in cluster_costs)
        if total_isolated_sats > 0:
            # 簇内不允许有孤星，直接扣满分
            score_penalty = 100
            
            isolated_details = []
            for cluster in cluster_costs:
                if cluster['isolated_count'] > 0:
                    isolated_sats_str = ', '.join(cluster['isolated_satellites'])
                    isolated_details.append(f"星簇{cluster['cluster_id']}内存在{isolated_sats_str}卫星是孤星")
            
            errors.append(f"[ERROR] 簇内不允许有孤星: {'; '.join(isolated_details)}，直接扣满分")
        else:
            # 只有在没有孤星的情况下才进行其他惩罚评估
            
            # 计算整个星座的总通信代价（所有卫星两两之间的连接代价）
            all_satellites = [attr.id for attr in conversation.input.sat_attrs]
            total_constellation_cost = 0
            
            for i, sat1 in enumerate(all_satellites):
                for sat2 in all_satellites[i+1:]:
                    # 直接使用sat_distances中的距离值，没有直接连接就不连接
                    distance = sat_distances.get((sat1, sat2), None)
                    if distance is not None:
                        total_constellation_cost += distance
                    # 如果没有直接连接，跳过这对卫星，不添加任何代价
            
            # 计算当前通信代价占整个星座总代价的比例
            if total_constellation_cost > 0:
                cost_ratio = total_cost / total_constellation_cost

                # 简化扣分计算：按比例直接映射到1-100分
                cost_penalty = min(int(cost_ratio * 100) + 1, 100)
                warnings.append(f"[WARNING] 通信代价占星座总代价比例: {cost_ratio:.1%}, 扣分{cost_penalty}分")
                score_penalty += cost_penalty
            else:
                # 如果星座总代价为0（异常情况），按原逻辑处理
                warnings.append("[WARNING] 无法计算星座总通信代价，跳过相对代价评估")

        # 确保扣分不超过100分
        score_penalty = min(score_penalty, 100)

        # 计算最终得分（满分100分减去扣分）
        final_score = int(100 - score_penalty)

        # 统计孤星信息
        total_isolated_sats = sum(cluster['isolated_count'] for cluster in cluster_costs)
        clusters_with_isolated = sum(1 for cluster in cluster_costs if cluster['isolated_count'] > 0)

        # 构建信息文本
        info_text = f"总通信代价: {total_cost:.1f}km, 簇内同步代价: {total_intra_cluster_cost:.1f}km, 全网同步代价: {total_inter_cluster_cost:.1f}km, 平均簇内代价: {avg_cluster_cost:.1f}km"
        
        # 添加星座总代价和比例信息（仅在没有孤星时显示）
        if total_isolated_sats == 0 and 'total_constellation_cost' in locals():
            cost_ratio = total_cost / total_constellation_cost if total_constellation_cost > 0 else 0
            info_text += f", 星座总代价: {total_constellation_cost:.1f}km, 代价比例: {cost_ratio:.1%}"
        
        if total_isolated_sats > 0:
            # 构建详细的孤星信息描述
            isolated_details = []
            for cluster in cluster_costs:
                if cluster['isolated_count'] > 0:
                    isolated_sats_str = ', '.join(cluster['isolated_satellites'])
                    isolated_details.append(f"星簇{cluster['cluster_id']}内存在{isolated_sats_str}卫星是孤星")
            info_text += f", 孤星详情: {'; '.join(isolated_details)}"
        info_text += f", 得分: {final_score}/100"
        
        if errors:
            info_text += f" | 错误: {'; '.join(errors)}"
        if warnings:
            info_text += f" | 警告: {'; '.join(warnings)}"

        return ValidationDetail(
            validation_type="communication_cost_validation",
            score=final_score,
            info=info_text,
        )

    def _validate_observation_efficiency_for_single_slice(
        self,
        conversation: LLMConversationMessage,
    ) -> ValidationDetail:
        """观测效能评估（100分）- 单个时间切片版本

        验证内容：
        1. 同一个簇内，目标被两颗卫星同时观测的概率

        Args:
            conversation: 单个对话数据

        Returns:
            ValidationDetail: 验证详情对象
        """
        errors = []
        warnings = []
        score_penalty = 0

        # 构建卫星到目标的连接映射
        sat_to_targets = defaultdict(set)
        target_to_sats = defaultdict(set)
        target_qualities = {}  # (sat_id, target_id) -> quality

        for edge in conversation.input.target_edges:
            sat_to_targets[edge.sat_id].add(edge.target_id)
            target_to_sats[edge.target_id].add(edge.sat_id)
            target_qualities[(edge.sat_id, edge.target_id)] = edge.quality

        total_dual_observation_probability = 0
        cluster_efficiency_stats = []

        for cluster_idx, cluster in enumerate(conversation.response.clusters):
            cluster_sats = set(cluster.sats)
            cluster_targets = set(cluster.targets)

            # 计算簇内目标被多颗卫星观测的情况
            dual_observed_targets = 0
            total_cluster_targets = len(cluster_targets)

            target_observation_details = []

            for target in cluster_targets:
                # 找到能观测该目标的簇内卫星
                observing_sats = target_to_sats[target] & cluster_sats

                if len(observing_sats) >= 2:
                    dual_observed_targets += 1
                    # 计算观测质量
                    qualities = [target_qualities.get((sat, target), 0) for sat in observing_sats]
                    avg_quality = sum(qualities) / len(qualities) if qualities else 0
                    target_observation_details.append({
                        'target': target,
                        'observing_sats': list(observing_sats),
                        'count': len(observing_sats),
                        'avg_quality': avg_quality
                    })
                elif len(observing_sats) == 1:
                    # 单卫星观测
                    sat = list(observing_sats)[0]
                    quality = target_qualities.get((sat, target), 0)
                    target_observation_details.append({
                        'target': target,
                        'observing_sats': list(observing_sats),
                        'count': 1,
                        'avg_quality': quality
                    })
                else:
                    # 无卫星观测（这种情况在正确性验证中应该已经被发现）
                    target_observation_details.append({
                        'target': target,
                        'observing_sats': [],
                        'count': 0,
                        'avg_quality': 0
                    })

            # 计算簇内双重观测概率
            dual_observation_rate = dual_observed_targets / total_cluster_targets if total_cluster_targets > 0 else 0

            # 计算簇内平均观测质量
            all_qualities = []
            for detail in target_observation_details:
                if detail['count'] > 0:
                    all_qualities.append(detail['avg_quality'])
            avg_cluster_quality = sum(all_qualities) / len(all_qualities) if all_qualities else 0

            cluster_efficiency_stats.append({
                'cluster_id': cluster_idx,
                'dual_observation_rate': dual_observation_rate,
                'avg_quality': avg_cluster_quality,
                'total_targets': total_cluster_targets,
                'dual_observed_targets': dual_observed_targets,
                'details': target_observation_details
            })

            total_dual_observation_probability += dual_observation_rate

        # 计算全局双重观测概率
        avg_dual_observation_rate = (total_dual_observation_probability / len(conversation.response.clusters)
                                   if conversation.response.clusters else 0)

        # 评估观测效能
        # 双重观测率过低的惩罚
        if avg_dual_observation_rate < 0.3:  # 低于30%的双重观测率
            efficiency_penalty = (0.3 - avg_dual_observation_rate) * 200  # 最多扣60分
            score_penalty += efficiency_penalty
            errors.append(f"[ERROR] 双重观测率过低: {avg_dual_observation_rate:.1%} < 30%")

        # 检查是否有簇完全没有双重观测
        zero_dual_clusters = [stat for stat in cluster_efficiency_stats if stat['dual_observation_rate'] == 0]
        if zero_dual_clusters:
            zero_penalty = len(zero_dual_clusters) * 10  # 每个无双重观测的簇扣10分
            score_penalty += zero_penalty
            warnings.append(f"[WARNING] 存在 {len(zero_dual_clusters)} 个簇无双重观测")

        # 检查观测质量
        all_cluster_qualities = [stat['avg_quality'] for stat in cluster_efficiency_stats if stat['avg_quality'] > 0]
        avg_global_quality = sum(all_cluster_qualities) / len(all_cluster_qualities) if all_cluster_qualities else 0

        if avg_global_quality < 0.5:  # 平均观测质量低于0.5
            quality_penalty = (0.5 - avg_global_quality) * 40  # 最多扣20分
            score_penalty += quality_penalty
            warnings.append(f"[WARNING] 平均观测质量过低: {avg_global_quality:.2f} < 0.5")

        # 确保扣分不超过100分
        score_penalty = min(score_penalty, 100)

        # 计算最终得分（满分100分减去扣分）
        final_score = int(100 - score_penalty)

        # 构建详细信息
        info_text = f"平均双重观测率: {avg_dual_observation_rate:.1%}, 平均观测质量: {avg_global_quality:.2f}, 得分: {final_score}/100"

        cluster_details = []
        for stat in cluster_efficiency_stats:
            cluster_details.append(
                f"簇{stat['cluster_id']}(双重观测率{stat['dual_observation_rate']:.1%}, 质量{stat['avg_quality']:.2f}, 目标数{stat['total_targets']})"
            )

        if cluster_details:
            info_text += f" | 簇详情: {'; '.join(cluster_details)}"
        if errors:
            info_text += f" | 错误: {'; '.join(errors)}"
        if warnings:
            info_text += f" | 警告: {'; '.join(warnings)}"

        return ValidationDetail(
            validation_type="observation_efficiency_validation",
            score=final_score,
            info=info_text,
        )

    def _validate_cluster_size_for_single_slice(
        self,
        conversation: LLMConversationMessage,
    ) -> ValidationDetail:
        """分簇规模验证（100分）- 单个时间切片版本

        验证内容：
        1. 小于等于2，大于等于10，都不合适

        Args:
            conversation: 单个对话数据

        Returns:
            ValidationDetail: 验证详情对象
        """
        errors = []
        warnings = []
        score_penalty = 0

        cluster_sizes = []
        size_distribution = {'too_small': 0, 'optimal': 0, 'too_large': 0}

        for cluster_idx, cluster in enumerate(conversation.response.clusters):
            cluster_size = len(cluster.sats)
            cluster_sizes.append(cluster_size)

            if cluster_size <= 2:
                size_distribution['too_small'] += 1
                errors.append(f"[ERROR] 簇 {cluster_idx} 规模过小: {cluster_size} 颗卫星")
                score_penalty += 20  # 每个过小的簇扣20分
            elif cluster_size >= 10:
                size_distribution['too_large'] += 1
                errors.append(f"[ERROR] 簇 {cluster_idx} 规模过大: {cluster_size} 颗卫星")
                score_penalty += 15  # 每个过大的簇扣15分
            else:
                size_distribution['optimal'] += 1

        # 计算规模统计信息
        total_clusters = len(conversation.response.clusters)
        avg_cluster_size = sum(cluster_sizes) / total_clusters if total_clusters > 0 else 0

        # 检查规模分布的合理性
        if size_distribution['too_small'] > total_clusters * 0.3:  # 超过30%的簇过小
            distribution_penalty = 20
            score_penalty += distribution_penalty
            warnings.append(f"[WARNING] 过小簇比例过高: {size_distribution['too_small']}/{total_clusters}")

        if size_distribution['too_large'] > total_clusters * 0.2:  # 超过20%的簇过大
            distribution_penalty = 15
            score_penalty += distribution_penalty
            warnings.append(f"[WARNING] 过大簇比例过高: {size_distribution['too_large']}/{total_clusters}")

        # 检查规模方差（规模分布是否均匀）
        if cluster_sizes:
            size_variance = sum((size - avg_cluster_size) ** 2 for size in cluster_sizes) / len(cluster_sizes)
            if size_variance > 9:  # 方差过大，说明规模分布不均
                variance_penalty = min(size_variance / 10 * 10, 10)  # 最多扣10分
                score_penalty += variance_penalty
                warnings.append(f"[WARNING] 簇规模分布不均: 方差 {size_variance:.1f}")

        # 确保扣分不超过100分
        score_penalty = min(score_penalty, 100)

        # 计算最终得分（满分100分减去扣分）
        final_score = int(100 - score_penalty)

        # 构建详细信息
        info_text = f"总簇数: {total_clusters}, 平均簇规模: {avg_cluster_size:.1f}颗卫星, 规模分布: 过小({size_distribution['too_small']}) | 合适({size_distribution['optimal']}) | 过大({size_distribution['too_large']}), 得分: {final_score}/100"

        if cluster_sizes:
            info_text += f", 最小簇规模: {min(cluster_sizes)}颗卫星, 最大簇规模: {max(cluster_sizes)}颗卫星"

        if errors:
            info_text += f" | 错误: {'; '.join(errors)}"
        if warnings:
            info_text += f" | 警告: {'; '.join(warnings)}"

        return ValidationDetail(
            validation_type="cluster_size_validation",
            score=final_score,
            info=info_text,
        )

    def _find_shortest_path_cost(self, start_sat, target_sat, 
                                sat_distances: Dict, cluster_sats):
        """在簇内找到从起始卫星到目标卫星的最短路径代价
        
        使用Dijkstra算法在簇内的卫星网络中寻找最短路径
        
        Args:
            start_sat: 起始卫星ID
            target_sat: 目标卫星ID（通常是主节点）
            sat_distances: 卫星间距离映射
            cluster_sats: 簇内所有卫星列表
            
        Returns:
            float: 最短路径代价，如果无法到达返回None
        """
        if start_sat == target_sat:
            return 0.0
            
        # 使用Dijkstra算法
        import heapq
        
        # 初始化距离和已访问集合
        distances = {sat: float('inf') for sat in cluster_sats}
        distances[start_sat] = 0.0
        visited = set()
        priority_queue = [(0.0, start_sat)]
        
        while priority_queue:
            current_distance, current_sat = heapq.heappop(priority_queue)
            
            if current_sat in visited:
                continue
                
            visited.add(current_sat)
            
            # 如果到达目标节点
            if current_sat == target_sat:
                return current_distance
            
            # 检查当前卫星的所有邻居
            for neighbor_sat in cluster_sats:
                if neighbor_sat in visited:
                    continue
                    
                # 检查是否有直接连接
                edge_distance = sat_distances.get((current_sat, neighbor_sat), None)
                if edge_distance is not None:
                    new_distance = current_distance + edge_distance
                    
                    if new_distance < distances[neighbor_sat]:
                        distances[neighbor_sat] = new_distance
                        heapq.heappush(priority_queue, (new_distance, neighbor_sat))
        
        # 无法到达目标节点
        return None
    
    def evaluate_all_results(self, validation_results: List[ValidationItem]) -> Dict[str, Any]:
        """评估所有结果

        Args:
            validation_results: 所有验证结果

        Returns:
            Dict[str, Any]: 详细的评估结果统计
        """
        if not validation_results:
            return {
                "total_samples": 0,
                "average_score": 0,
                "max_score": 0,
                "min_score": 0,
                "score_distribution": {},
                "validation_type_stats": {},
                "loss_item_reasons": [],
                "perfect_score_count": 0,
                "perfect_score_rate": 0
            }

        # 定义权重（用于显示和统计，实际计算在ValidationItem.score中）
        weights = {
            "correctness_validation": 0.4,      # 正确性和隔离性 40%
            "stability_validation": 0.3,        # 分簇稳定性 30%
            "communication_cost_validation": 0.1,   # 通信代价 10%
            "observation_efficiency_validation": 0.1,  # 观测效能 10%
            "cluster_size_validation": 0.1      # 分簇规模 10%
        }

        # 基础统计
        total_samples = len(validation_results)
        all_scores = []
        
        # 使用ValidationItem的score属性，该属性已经包含加权计算
        for item in validation_results:
            all_scores.append(int(item.score))
        
        total_score = sum(all_scores)
        average_score = total_score / total_samples
        max_score = max(all_scores)
        min_score = min(all_scores)

        # 满分统计（总分100分）
        perfect_scores = [score for score in all_scores if score == 100]
        perfect_score_count = len(perfect_scores)
        perfect_score_rate = perfect_score_count / total_samples

        # 分数分布统计
        score_ranges = {
            "90-100": 0,
            "80-89": 0,
            "70-79": 0,
            "60-69": 0,
            "50-59": 0,
            "40-49": 0,
            "30-39": 0,
            "20-29": 0,
            "10-19": 0,
            "0-9": 0
        }

        for score in all_scores:
            if 90 <= score <= 100:
                score_ranges["90-100"] += 1
            elif 80 <= score < 90:
                score_ranges["80-89"] += 1
            elif 70 <= score < 80:
                score_ranges["70-79"] += 1
            elif 60 <= score < 70:
                score_ranges["60-69"] += 1
            elif 50 <= score < 60:
                score_ranges["50-59"] += 1
            elif 40 <= score < 50:
                score_ranges["40-49"] += 1
            elif 30 <= score < 40:
                score_ranges["30-39"] += 1
            elif 20 <= score < 30:
                score_ranges["20-29"] += 1
            elif 10 <= score < 20:
                score_ranges["10-19"] += 1
            else:
                score_ranges["0-9"] += 1

        # 各验证类型的详细统计
        validation_type_stats = {
            "correctness_validation": {"scores": [], "max_score": 100, "avg_score": 0, "loss_count": 0, "loss_reasons": []},
            "stability_validation": {"scores": [], "max_score": 100, "avg_score": 0, "loss_count": 0, "loss_reasons": []},
            "communication_cost_validation": {"scores": [], "max_score": 100, "avg_score": 0, "loss_count": 0, "loss_reasons": []},
            "observation_efficiency_validation": {"scores": [], "max_score": 100, "avg_score": 0, "loss_count": 0, "loss_reasons": []},
            "cluster_size_validation": {"scores": [], "max_score": 100, "avg_score": 0, "loss_count": 0, "loss_reasons": []}
        }

        # 收集丢分项目和原因
        loss_item_reasons = []

        for item_idx, item in enumerate(validation_results):
            for detail in item.validation_details:
                validation_type = detail.validation_type
                score = detail.score
                max_possible = validation_type_stats[validation_type]["max_score"]

                # 记录分数
                validation_type_stats[validation_type]["scores"].append(score)

                # 检查是否丢分
                if score < max_possible:
                    loss_points = max_possible - score
                    validation_type_stats[validation_type]["loss_count"] += 1

                    # 提取丢分原因（从info字段中提取错误和警告）
                    info = detail.info
                    reasons = []
                    if "错误:" in info:
                        error_part = info.split("错误:")[1].split("|")[0].strip()
                        reasons.append(f"错误: {error_part}")
                    if "警告:" in info:
                        warning_part = info.split("警告:")[1].strip()
                        reasons.append(f"警告: {warning_part}")

                    loss_reason = {
                        "sample_index": item_idx,
                        "validation_type": validation_type,
                        "score": score,
                        "max_score": max_possible,
                        "loss_points": loss_points,
                        "reasons": reasons if reasons else ["未知原因"],
                        "full_info": info
                    }
                    loss_item_reasons.append(loss_reason)
                    validation_type_stats[validation_type]["loss_reasons"].append(loss_reason)

        # 计算各验证类型的平均分
        for validation_type, stats in validation_type_stats.items():
            if stats["scores"]:
                stats["avg_score"] = sum(stats["scores"]) / len(stats["scores"])
                stats["min_score"] = min(stats["scores"])
                stats["max_score_achieved"] = max(stats["scores"])
                stats["loss_rate"] = stats["loss_count"] / len(stats["scores"])
            else:
                stats["avg_score"] = 0
                stats["min_score"] = 0
                stats["max_score_achieved"] = 0
                stats["loss_rate"] = 0

        return {
            "total_samples": total_samples,
            "average_score": round(average_score, 2),
            "max_score": max_score,
            "min_score": min_score,
            "score_distribution": score_ranges,
            "validation_type_stats": validation_type_stats,
            "loss_item_reasons": loss_item_reasons,
            "total_loss_items": len(loss_item_reasons),
            "perfect_score_count": perfect_score_count,
            "perfect_score_rate": round(perfect_score_rate, 4),
            "summary": {
                "total_samples": total_samples,
                "average_score": round(average_score, 2),
                "score_range": f"{min_score}-{max_score}",
                "perfect_rate": f"{perfect_score_rate:.1%}",
                "most_common_loss_type": self._get_most_common_loss_type(loss_item_reasons)
            }
        }

    def _get_most_common_loss_type(self, loss_items: List[Dict]) -> str:
        """获取最常见的丢分类型"""
        if not loss_items:
            return "无丢分"

        type_counts = {}
        for item in loss_items:
            validation_type = item["validation_type"]
            type_counts[validation_type] = type_counts.get(validation_type, 0) + 1

        most_common = max(type_counts.items(), key=lambda x: x[1])
        return f"{most_common[0]} ({most_common[1]}次)"

    def print_evaluation_summary(self, evaluation_result: Dict[str, Any]) -> None:
        """打印评估结果摘要

        Args:
            evaluation_result: evaluate_all_results返回的结果
        """
        print("=" * 80)
        print("📊 卫星分簇验证结果统计报告")
        print("=" * 80)
        print("🔄 评分体系说明:")
        print("   每项验证采用100分制，通过加权求和计算总分：")
        print("   • 正确性验证: 100分 × 40% = 40分")
        print("   • 分簇稳定性: 100分 × 30% = 30分") 
        print("   • 通信代价: 100分 × 10% = 10分")
        print("   • 观测效能: 100分 × 10% = 10分")
        print("   • 分簇规模: 100分 × 10% = 10分")
        print("   总分 = 各项得分 × 对应权重之和")
        print("=" * 80)

        summary = evaluation_result["summary"]
        print(f"📈 总体统计:")
        print(f"   样本总数: {summary['total_samples']}")
        print(f"   平均分数: {summary['average_score']}/100")
        print(f"   分数范围: {summary['score_range']}")
        print(f"   满分率: {summary['perfect_rate']}")
        print(f"   最常见丢分类型: {summary['most_common_loss_type']}")

        print(f"\n📊 分数分布:")
        for score_range, count in evaluation_result["score_distribution"].items():
            if count > 0:
                percentage = count / evaluation_result["total_samples"] * 100
                print(f"   {score_range}分: {count}个样本 ({percentage:.1f}%)")

        print(f"\n🔍 各验证类型详细统计:")
        type_names = {
            "correctness_validation": "正确性验证(100分)",
            "stability_validation": "分簇稳定性(100分)",
            "communication_cost_validation": "通信代价(100分)",
            "observation_efficiency_validation": "观测效能(100分)",
            "cluster_size_validation": "分簇规模(100分)"
        }

        for validation_type, stats in evaluation_result["validation_type_stats"].items():
            type_name = type_names.get(validation_type, validation_type)
            print(f"   {type_name}:")
            print(f"     平均分: {stats['avg_score']:.1f}/{stats['max_score']}")
            print(f"     分数范围: {stats['min_score']}-{stats['max_score_achieved']}")
            print(f"     丢分率: {stats['loss_rate']:.1%} ({stats['loss_count']}个样本)")

        print(f"\n⚠️  丢分项目统计 (共{evaluation_result['total_loss_items']}项):")
        if evaluation_result["loss_item_reasons"]:
            # 按验证类型分组显示丢分原因
            loss_by_type = {}
            for loss_item in evaluation_result["loss_item_reasons"]:
                validation_type = loss_item["validation_type"]
                if validation_type not in loss_by_type:
                    loss_by_type[validation_type] = []
                loss_by_type[validation_type].append(loss_item)

            for validation_type, losses in loss_by_type.items():
                type_name = type_names.get(validation_type, validation_type)
                print(f"   {type_name} ({len(losses)}项丢分):")

                # 统计相同原因的丢分
                reason_counts = {}
                for loss in losses:
                    for reason in loss["reasons"]:
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1

                for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"     - {reason}: {count}次")
        else:
            print("   🎉 没有丢分项目！")

        print("=" * 80)


if __name__ == "__main__":
    timestamp = get_current_timestamp()
    input_file_name = "training_data/training_data_sharegpt_gemini-2.5-pro_20250629_103625_30_v3.1_simplified.jsonl"
    data_path = get_data_dir() / input_file_name
    raw_data: List[LLMConversationMessage] = load_sharegpt_data(data_path)
    validator = ClusterDataValidator()

    print("🚀 开始验证卫星分簇结果...")
    validation_results = validator.validate_output(raw_data)

    print("📊 生成详细统计报告...")
    evaluation_result = validator.evaluate_all_results(validation_results)

    # 打印统计摘要
    validator.print_evaluation_summary(evaluation_result)

    # 保存验证结果
    output_file = get_data_dir() / f"training_data_sharegpt_gemini-2.5-pro_20250629_103625_30_v3_validation_result_{timestamp}.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in validation_results:
            f.write(json.dumps(item, default=lambda o: o.__dict__, ensure_ascii=False) + "\n")

    # 保存统计结果
    stats_file = get_data_dir() / f"{input_file_name}_validation_stats_{timestamp}.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_result, f, ensure_ascii=False, indent=2)

    print(f"✅ 验证完成！结果已保存到:")
    print(f"   验证结果: {output_file}")
    print(f"   统计报告: {stats_file}")
