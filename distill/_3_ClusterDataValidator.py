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
        
        for edge in conversation.input.sat_edges:
            input_satellites.add(edge.from_sat)
            input_satellites.add(edge.to_sat)
        
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
            # 发现致命错误，直接扣满分并标记error错误
            error_details = []
            
            if invalid_targets:
                target_list = sorted(list(invalid_targets))
                error_details.append(f"不存在的目标: {', '.join(map(str, target_list))}")
            
            if invalid_satellites:
                sat_list = sorted(list(invalid_satellites))
                error_details.append(f"不存在的卫星: {', '.join(map(str, sat_list))}")
            
            error_msg = f"[ERROR] 分簇中包含输入中不存在的元素: {'; '.join(error_details)}"
            
            return ValidationDetail(
                validation_type="correctness_validation",
                score=0,  # 直接扣满分
                info=error_msg,
            )
        
        # === 2. 目标遗漏检测和扣分（最多50分）===
        missing_targets = input_targets - output_targets
        target_coverage_rate = len(output_targets & input_targets) / len(input_targets) if input_targets else 1.0
        
        score_penalty = 0
        errors = []
        warnings = []
        
        if missing_targets:
            coverage_penalty = (1.0 - target_coverage_rate) * 50
            score_penalty += coverage_penalty
            
            # 构建遗漏目标详情 - 显示全部
            missing_list = sorted(list(missing_targets))
            displayed_targets = missing_list
            
            warnings.append(f"[WARNING] 目标遗漏{len(missing_targets)}个，扣{coverage_penalty:.1f}分：{', '.join(map(str, displayed_targets))}")
        
        # === 3. 隔离性验证和扣分（最多50分）===
        # 3.1 检查卫星-目标连接跨簇
        cross_cluster_violations = []
        total_valid_connections = 0
        
        for sat, target in valid_sat_target_connections:
            sat_cluster = satellite_cluster_map.get(sat)
            target_cluster = target_cluster_map.get(target)
            
            if sat_cluster is not None and target_cluster is not None:
                total_valid_connections += 1
                if sat_cluster != target_cluster:
                    cross_cluster_violations.append({'satellite': sat, 'target': target})
        
        # 3.2 检查多簇归属
        satellite_to_clusters = defaultdict(list)
        target_to_clusters = defaultdict(list)
        
        for cluster_idx, cluster in enumerate(conversation.response.clusters):
            for satellite in cluster.sats:
                satellite_to_clusters[satellite].append(cluster_idx)
            for target in cluster.targets:
                target_to_clusters[target].append(cluster_idx)
        
        satellite_multi_cluster_violations = [sat for sat, clusters in satellite_to_clusters.items() if len(clusters) > 1]
        target_multi_cluster_violations = [target for target, clusters in target_to_clusters.items() if len(clusters) > 1]
        
        # 隔离性扣分计算
        isolation_penalty = 0
        
        # 连接跨簇扣分（最多25分）
        if cross_cluster_violations and total_valid_connections > 0:
            cross_cluster_rate = len(cross_cluster_violations) / total_valid_connections
            connection_penalty = cross_cluster_rate * 25
            isolation_penalty += connection_penalty
            
            # 构建跨簇连接详情 - 显示全部
            violation_details = []
            for violation in cross_cluster_violations:
                sat = violation['satellite']
                target = violation['target']
                sat_cluster = satellite_cluster_map.get(sat)
                target_cluster = target_cluster_map.get(target)
                violation_details.append(f"卫星{sat}(簇{sat_cluster})-目标{target}(簇{target_cluster})")
            
            warnings.append(f"[WARNING] 连接跨簇{len(cross_cluster_violations)}个，扣{connection_penalty:.1f}分：{'; '.join(violation_details)}")
        
        # 多簇归属扣分（最多25分）
        if satellite_multi_cluster_violations:
            satellite_penalty = min(len(satellite_multi_cluster_violations) * 5, 12.5)
            isolation_penalty += satellite_penalty
            
            # 构建卫星多簇详情 - 显示全部
            sat_details = []
            for sat in satellite_multi_cluster_violations:
                clusters = satellite_to_clusters[sat]
                sat_details.append(f"卫星{sat}(簇{clusters})")
            
            warnings.append(f"[WARNING] 卫星多簇{len(satellite_multi_cluster_violations)}个，扣{satellite_penalty:.1f}分：{'; '.join(sat_details)}")
        
        if target_multi_cluster_violations:
            target_penalty = min(len(target_multi_cluster_violations) * 5, 12.5)
            isolation_penalty += target_penalty
            
            # 构建目标多簇详情 - 显示全部
            target_details = []
            for target in target_multi_cluster_violations:
                clusters = target_to_clusters[target]
                target_details.append(f"目标{target}(簇{clusters})")
            
            warnings.append(f"[WARNING] 目标多簇{len(target_multi_cluster_violations)}个，扣{target_penalty:.1f}分：{'; '.join(target_details)}")
        
        # 累加隔离性扣分到总扣分
        score_penalty += isolation_penalty
        
        # 确保扣分不超过100分
        score_penalty = min(score_penalty, 100)
        final_score = int(100 - score_penalty)
        
        # 构建格式化信息文本
        cross_cluster_rate = len(cross_cluster_violations) / total_valid_connections if total_valid_connections > 0 else 0
        
        # 构建详细信息
        info_parts = []
        
        # 添加错误信息
        if errors:
            for error in errors:
                info_parts.append(error)
        
        # 添加警告信息  
        if warnings:
            for warning in warnings:
                info_parts.append(warning)
        
        # 添加摘要信息
        summary = (f"[SUMMARY] 覆盖率:{target_coverage_rate:.1%}, 跨簇率:{cross_cluster_rate:.1%}, "
                  f"多簇卫星:{len(satellite_multi_cluster_violations)}个, 多簇目标:{len(target_multi_cluster_violations)}个, "
                  f"得分:{final_score}/100")
        info_parts.append(summary)
        
        info_text = "\n".join(info_parts)

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
        1. 历史数据检查：无历史数据时跳过验证 → 满分100分
        2. 目标稳定性检查：目标在不同时间切片间的簇归属变化 → 按切换率扣分（最多50分）
           - 豁免场景：如果目标已无法被原簇观测，则切换不扣分。
        3. 卫星稳定性检查：卫星在不同时间切片间的簇归属变化 → 按切换率扣分（最多30分）
           - 豁免场景：如果卫星在原簇已成孤岛，则切换不扣分。
        4. Jaccard相似度检查：前后分簇的重叠率 → 低于80%扣分（最多20分）

        Args:
            conversation: 单个对话数据

        Returns:
            ValidationDetail: 验证详情对象
        """
        # 初始化验证结果
        score_penalty = 0
        errors = []
        warnings = []
        info_logs = []

        # === 1. 历史数据有效性检查 ===
        history_results = conversation.input.history_cluster_result
        if not history_results or len(history_results) == 0:
            # 没有历史数据，无法进行稳定性验证，直接给满分
            return ValidationDetail(
                validation_type="stability_validation",
                score=100,
                info="[INFO] 无历史分簇数据，跳过稳定性验证",
            )

        # 获取最近一次的历史分簇结果
        last_clusters = history_results[-1] if history_results else []
        current_clusters = conversation.response.clusters

        if not last_clusters:
            # 历史数据为空，给0分并警告
            return ValidationDetail(
                validation_type="stability_validation",
                score=0,
                info="[ERROR] 历史分簇数据为空，无法进行稳定性验证",
            )

        # === 2. 构建当前和历史分簇的映射关系 ===
        # 构建历史分簇的目标和卫星映射
        last_target_to_cluster = {}
        last_satellite_to_cluster = {}
        last_cluster_sats = defaultdict(list)

        for cluster_idx, cluster in enumerate(last_clusters):
            for target in cluster.targets:
                last_target_to_cluster[target] = cluster_idx
            for satellite in cluster.sats:
                last_satellite_to_cluster[satellite] = cluster_idx
                last_cluster_sats[cluster_idx].append(satellite)

        # 构建当前分簇的目标和卫星映射
        current_target_to_cluster = {}
        current_satellite_to_cluster = {}

        for cluster_idx, cluster in enumerate(current_clusters):
            for target in cluster.targets:
                current_target_to_cluster[target] = cluster_idx
            for satellite in cluster.sats:
                current_satellite_to_cluster[satellite] = cluster_idx
        
        # 构建当前时间切片的连接关系以供豁免检查
        # 1. 目标可见性：哪些卫星能看到哪些目标
        target_visibility = defaultdict(set)
        for edge in conversation.input.target_edges:
            target_visibility[edge.target_id].add(edge.sat_id)
        
        # 2. 卫星连通性：哪些卫星之间有连接
        sat_connectivity = defaultdict(set)
        for edge in conversation.input.sat_edges:
            sat_connectivity[edge.from_sat].add(edge.to_sat)
            sat_connectivity[edge.to_sat].add(edge.from_sat)

        # === 3. 目标稳定性验证和扣分（最多50分）===
        target_switches = []
        justified_target_switches = 0
        common_targets = set(last_target_to_cluster.keys()) & set(current_target_to_cluster.keys())
        total_targets = len(set(last_target_to_cluster.keys()) | set(current_target_to_cluster.keys()))

        for target in common_targets:
            last_cluster_idx = last_target_to_cluster[target]
            current_cluster_idx = current_target_to_cluster[target]
            
            if last_cluster_idx != current_cluster_idx:
                # 检查豁免条件：目标是否已无法被原簇观测
                sats_in_last_cluster = set(last_cluster_sats.get(last_cluster_idx, []))
                visible_sats_for_target = target_visibility.get(target, set())
                
                # 如果目标在当前时间切片，与旧簇的卫星已无任何可见关系
                if not sats_in_last_cluster.intersection(visible_sats_for_target):
                    justified_target_switches += 1
                    info_logs.append(f"[INFO] 目标{target}切换被豁免：已无法被原簇{last_cluster_idx}中的任何卫星观测")
                    continue  # 跳过，不计入惩罚

                target_switches.append({
                    'target': target,
                    'from_cluster': last_cluster_idx,
                    'to_cluster': current_cluster_idx
                })

        target_switch_count = len(target_switches)
        target_switch_rate = target_switch_count / total_targets if total_targets > 0 else 0

        # 目标切换惩罚
        if target_switch_rate > 0.2:  # 超过20%的目标切换
            target_penalty = min(target_switch_rate * 100, 50)
            score_penalty += target_penalty
            
            # 构建目标切换详情 - 显示全部
            switch_details = []
            for switch in target_switches:
                switch_details.append(f"目标{switch['target']}(簇{switch['from_cluster']}→簇{switch['to_cluster']})")
            
            warnings.append(f"[WARNING] 目标切换{target_switch_count}个，扣{target_penalty:.1f}分：{'; '.join(switch_details)}")

        # === 4. 卫星稳定性验证和扣分（最多30分）===
        satellite_switches = []
        justified_satellite_switches = 0
        common_satellites = set(last_satellite_to_cluster.keys()) & set(current_satellite_to_cluster.keys())
        total_satellites = len(set(last_satellite_to_cluster.keys()) | set(current_satellite_to_cluster.keys()))

        for satellite in common_satellites:
            last_cluster_idx = last_satellite_to_cluster[satellite]
            current_cluster_idx = current_satellite_to_cluster[satellite]

            if last_cluster_idx != current_cluster_idx:
                # 检查豁免条件：卫星是否在原簇已成孤岛
                sats_in_last_cluster = set(last_cluster_sats.get(last_cluster_idx, []))
                sats_in_last_cluster.discard(satellite) # 排除自己
                
                connected_sats = sat_connectivity.get(satellite, set())
                
                # 如果卫星在当前时间切片，与旧簇的其他卫星已无任何连接
                if not sats_in_last_cluster.intersection(connected_sats):
                    justified_satellite_switches += 1
                    info_logs.append(f"[INFO] 卫星{satellite}切换被豁免：在原簇{last_cluster_idx}中已成孤星")
                    continue # 跳过，不计入惩罚

                satellite_switches.append({
                    'satellite': satellite,
                    'from_cluster': last_cluster_idx,
                    'to_cluster': current_cluster_idx
                })

        satellite_switch_count = len(satellite_switches)
        satellite_switch_rate = satellite_switch_count / total_satellites if total_satellites > 0 else 0

        # 卫星切换惩罚
        if satellite_switch_rate > 0.15:  # 超过15%的卫星切换
            satellite_penalty = min(satellite_switch_rate * 100, 30) # 按比例扣分，最多30分
            score_penalty += satellite_penalty
            
            # 构建卫星切换详情 - 显示全部
            switch_details = []
            for switch in satellite_switches:
                switch_details.append(f"卫星{switch['satellite']}(簇{switch['from_cluster']}→簇{switch['to_cluster']})")
            
            warnings.append(f"[WARNING] 卫星切换{satellite_switch_count}个，扣{satellite_penalty:.1f}分：{'; '.join(switch_details)}")

        # === 5. Jaccard相似度验证和扣分（最多20分）===
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

        # Jaccard相似度惩罚
        jaccard_penalty = 0
        if avg_jaccard < 0.8:
            unjustified_switches = target_switch_count + satellite_switch_count
            total_switches = unjustified_switches + justified_target_switches + justified_satellite_switches

            if unjustified_switches > 0 and total_switches > 0:
                # 计算潜在的Jaccard惩罚
                potential_jaccard_penalty = (1 - (avg_jaccard / 0.8)) * 20
                
                # 根据不合理切换的比例来缩放惩罚
                unjustified_ratio = unjustified_switches / total_switches
                jaccard_penalty = potential_jaccard_penalty * unjustified_ratio
                
                warnings.append(f"[WARNING] 簇重叠率{avg_jaccard:.1%}低于80%阈值，因不合理切换占比{unjustified_ratio:.1%}，扣{jaccard_penalty:.1f}分")
            elif unjustified_switches == 0:
                info_logs.append(f"[INFO] Jaccard相似度惩罚被豁免：所有成员切换均为合理调整，虽然簇重叠率({avg_jaccard:.1%})较低，但不扣分。")

        score_penalty += jaccard_penalty

        # 确保扣分不超过100分
        score_penalty = min(score_penalty, 100)
        final_score = int(100 - score_penalty)

        # 构建详细信息
        info_parts = []
        
        # 添加豁免信息
        if info_logs:
            for log in info_logs:
                info_parts.append(log)

        # 添加警告信息
        if warnings:
            for warning in warnings:
                info_parts.append(warning)
        
        # 添加摘要信息
        summary = (f"[SUMMARY] 目标切换率:{target_switch_rate:.1%}({target_switch_count}/{total_targets}, 豁免{justified_target_switches}个), "
                  f"卫星切换率:{satellite_switch_rate:.1%}({satellite_switch_count}/{total_satellites}, 豁免{justified_satellite_switches}个), "
                  f"簇重叠率:{avg_jaccard:.1%}, 得分:{final_score}/100")
        info_parts.append(summary)
        
        info_text = "\n".join(info_parts)

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
        1. 致命错误检测：主节点不在簇内、簇内存在孤星 → 扣全部100分，标记ERROR错误
        2. 通信代价评估：分析簇内同步代价和全网同步代价 → 按代价比例扣分（最多100分）
        
        扣分机制：
        - 主节点无效或孤星存在：直接扣满分100分
        - 通信代价过高：按占总星座代价比例映射扣分

        Args:
            conversation: 单个对话数据

        Returns:
            ValidationDetail: 验证详情对象
        """
        # 构建卫星基础信息
        sat_positions = {}
        sat_distances = {}
        all_satellites = set()
        
        for sat_attr in conversation.input.sat_attrs:
            sat_positions[sat_attr.id] = sat_attr.pos
            all_satellites.add(sat_attr.id)

        for edge in conversation.input.sat_edges:
            sat_distances[(edge.from_sat, edge.to_sat)] = edge.distance
            sat_distances[(edge.to_sat, edge.from_sat)] = edge.distance  # 双向距离

        # 初始化验证结果
        score_penalty = 0
        errors = []
        warnings = []
        
        # === 1. 致命错误检测：主节点有效性和连通性 ===
        invalid_master_clusters = []
        isolated_satellite_details = []
        
        for cluster_idx, cluster in enumerate(conversation.response.clusters):
            cluster_sats = cluster.sats
            master_sat = cluster.master

            # 检查主节点是否在簇内卫星列表中
            if master_sat not in cluster_sats:
                invalid_master_clusters.append({'cluster_id': cluster_idx, 'master': master_sat, 'sats': cluster_sats})
                continue

            # 检查簇内卫星到主节点的连通性
            isolated_satellites = []
            for member_sat in cluster_sats:
                if member_sat == master_sat:
                    continue  # 主节点不需要检查到自己的连通性
                
                path_cost = self._find_shortest_path_cost(
                    member_sat, master_sat, sat_distances, cluster_sats
                )
                
                if path_cost is None:
                    isolated_satellites.append(member_sat)
            
            if isolated_satellites:
                isolated_satellite_details.append({
                    'cluster_id': cluster_idx,
                    'master': master_sat,
                    'isolated_sats': isolated_satellites
                })

        # 如果存在致命错误，直接扣满分
        if invalid_master_clusters or isolated_satellite_details:
            error_details = []
            
            if invalid_master_clusters:
                master_errors = []
                for cluster_info in invalid_master_clusters:
                    master_errors.append(f"簇{cluster_info['cluster_id']}主节点{cluster_info['master']}不在簇内")
                error_details.append(f"主节点无效: {'; '.join(master_errors)}")
            
            if isolated_satellite_details:
                isolated_errors = []
                for detail in isolated_satellite_details:
                    isolated_sats_str = ', '.join(map(str, detail['isolated_sats']))
                    isolated_errors.append(f"簇{detail['cluster_id']}内卫星{isolated_sats_str}无法连通主节点{detail['master']}")
                error_details.append(f"孤星存在: {'; '.join(isolated_errors)}")
            
            error_msg = f"[ERROR] 通信网络存在致命问题: {'; '.join(error_details)}"
            
            return ValidationDetail(
                validation_type="communication_cost_validation",
                score=0,  # 直接扣满分
                info=error_msg,
            )

        # === 2. 通信代价计算和评估（最多100分）===
        total_intra_cluster_cost = 0
        total_inter_cluster_cost = 0
        cluster_cost_details = []

        # 计算簇内同步代价
        for cluster_idx, cluster in enumerate(conversation.response.clusters):
            cluster_sats = cluster.sats
            master_sat = cluster.master
            intra_cluster_cost = 0
            
            for member_sat in cluster_sats:
                if member_sat == master_sat:
                    continue
                
                path_cost = self._find_shortest_path_cost(
                    member_sat, master_sat, sat_distances, cluster_sats
                )
                
                if path_cost is not None:
                    intra_cluster_cost += path_cost

            total_intra_cluster_cost += intra_cluster_cost
            cluster_cost_details.append({
                'cluster_id': cluster_idx,
                'intra_cost': intra_cluster_cost,
                'master': master_sat,
                'size': len(cluster_sats)
            })

        # 计算全网同步代价（主节点间通信）
        masters = [cluster.master for cluster in conversation.response.clusters]
        
        for i, master1 in enumerate(masters):
            for master2 in masters[i+1:]:
                path_cost = self._find_shortest_path_cost(
                    master1, master2, sat_distances, all_satellites
                )
                
                if path_cost is not None:
                    total_inter_cluster_cost += path_cost
                # 如果无法连通，则不计算代价（忽略此连接）

        total_cost = total_intra_cluster_cost + total_inter_cluster_cost

        # 计算星座总通信代价作为基准（所有可联通的卫星对之间的最短路径代价总和）
        total_constellation_cost = 0
        for i, sat1 in enumerate(all_satellites):
            for sat2 in list(all_satellites)[i+1:]:
                # 计算任意两颗卫星之间的最短路径代价
                path_cost = self._find_shortest_path_cost(
                    sat1, sat2, sat_distances, all_satellites
                )
                if path_cost is not None:
                    total_constellation_cost += path_cost

        # 通信效率评估（按比例扣分）
        cost_penalty = 0
        if total_constellation_cost > 0:
            cost_ratio = total_cost / total_constellation_cost
            efficiency_improvement = (1 - cost_ratio) * 100  # 效率提升百分比
            
            # 按代价比例扣分：代价比例越高，扣分越多
            if cost_ratio > 0.1:  # 超过10%的星座总代价开始扣分
                # 线性扣分：10%时扣0分，100%时扣满100分
                cost_penalty = min((cost_ratio - 0.1) / 0.9 * 100, 100)
                warnings.append(f"[WARNING] 分了{len(conversation.response.clusters)}个簇后通信代价仍然很高，扣{cost_penalty:.1f}分：当前总通信距离{total_cost:.1f}km，是全星座通信距离的{cost_ratio:.1%}，只节省了{efficiency_improvement:.1f}%的通信成本")
            else:
                warnings.append(f"[INFO] 分了{len(conversation.response.clusters)}个簇后通信效率很好，不扣分：当前总通信距离{total_cost:.1f}km，仅占全星座通信距离的{cost_ratio:.1%}，节省了{efficiency_improvement:.1f}%的通信成本")
        else:
            warnings.append("[WARNING] 无法计算星座基准代价，跳过效率评估")

        score_penalty += cost_penalty
        
        # 确保扣分不超过100分
        score_penalty = min(score_penalty, 100)
        final_score = int(100 - score_penalty)
        
        # 构建详细信息
        info_parts = []
        
        # 添加警告信息
        if warnings:
            for warning in warnings:
                info_parts.append(warning)
        
        # 添加摘要信息
        avg_cluster_cost = total_intra_cluster_cost / len(conversation.response.clusters) if conversation.response.clusters else 0
        cost_ratio = total_cost / total_constellation_cost if total_constellation_cost > 0 else 0
        
        summary = (f"[SUMMARY] 总代价:{total_cost:.1f}km, 簇内:{total_intra_cluster_cost:.1f}km, "
                  f"全网:{total_inter_cluster_cost:.1f}km, 平均簇内:{avg_cluster_cost:.1f}km, "
                  f"代价比例:{cost_ratio:.1%}, 得分:{final_score}/100")
        info_parts.append(summary)
        
        info_text = "\n".join(info_parts)

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
                variance_penalty = min(size_variance / 10 * 10, 10) # 最多扣10分
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
