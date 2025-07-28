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

        验证体系（每项100分，加权求和计算最终得分）：

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

        3、通信代价（100分，权重15%）:
            簇内同步代价：1x distance
            全网同步代价：1.2x master node distance，涉及到主节点的选择。

        4、观测效能评估（100分，权重15%）：
            同一个簇内，目标被两颗卫星同时观测的概率

        最终得分 = 正确性×0.4 + 稳定性×0.3 + 通信代价×0.15 + 观测效能×0.15

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
                correctness_detail = (
                    self._validate_correctness_and_isolation_for_single_slice(
                        conversation
                    )
                )
                validation_details.append(correctness_detail)

                # 2. 分簇稳定性验证（100分，权重30%）
                stability_detail = self._validate_stability_for_single_slice(
                    conversation
                )
                validation_details.append(stability_detail)

                # 3. 通信代价验证（100分，权重15%）
                cost_detail = self._validate_communication_cost_for_single_slice(
                    conversation
                )
                validation_details.append(cost_detail)

                # 4. 观测效能验证（100分，权重15%）
                efficiency_detail = (
                    self._validate_observation_efficiency_for_single_slice(conversation)
                )
                validation_details.append(efficiency_detail)

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
        invalid_targets = (
            output_targets - input_targets
        )  # 输出中存在但输入中不存在的目标
        invalid_satellites = (
            output_satellites - input_satellites
        )  # 输出中存在但输入中不存在的卫星

        if invalid_targets or invalid_satellites:
            # 发现致命错误，直接扣满分并标记error错误
            error_details = []

            if invalid_targets:
                target_list = sorted(list(invalid_targets))
                error_details.append(
                    f"不存在的目标: {', '.join(map(str, target_list))}"
                )

            if invalid_satellites:
                sat_list = sorted(list(invalid_satellites))
                error_details.append(f"不存在的卫星: {', '.join(map(str, sat_list))}")

            error_msg = (
                f"[ERROR] 分簇中包含输入中不存在的元素: {'; '.join(error_details)}"
            )

            return ValidationDetail(
                validation_type="correctness_validation",
                score=0,  # 直接扣满分
                info=error_msg,
            )

        # === 2. 目标遗漏检测和扣分（致命错误）===
        missing_targets = input_targets - output_targets
        if missing_targets:
            missing_list = sorted(list(missing_targets))
            error_msg = f"[ERROR] 目标遗漏{len(missing_list)}个: {', '.join(map(str, missing_list))}"
            return ValidationDetail(
                validation_type="correctness_validation",
                score=0,  # 直接扣满分
                info=error_msg,
            )

        # 如果没有目标遗漏，则覆盖率为100%
        target_coverage_rate = 1.0

        score_penalty = 0
        errors = []
        warnings = []

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
                    cross_cluster_violations.append(
                        {"satellite": sat, "target": target}
                    )

        # 3.2 检查多簇归属
        satellite_to_clusters = defaultdict(list)
        target_to_clusters = defaultdict(list)

        for cluster_idx, cluster in enumerate(conversation.response.clusters):
            for satellite in cluster.sats:
                satellite_to_clusters[satellite].append(cluster_idx)
            for target in cluster.targets:
                target_to_clusters[target].append(cluster_idx)

        satellite_multi_cluster_violations = [
            sat for sat, clusters in satellite_to_clusters.items() if len(clusters) > 1
        ]
        target_multi_cluster_violations = [
            target
            for target, clusters in target_to_clusters.items()
            if len(clusters) > 1
        ]

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
                sat = violation["satellite"]
                target = violation["target"]
                sat_cluster = satellite_cluster_map.get(sat)
                target_cluster = target_cluster_map.get(target)
                violation_details.append(
                    f"卫星{sat}(簇{sat_cluster})-目标{target}(簇{target_cluster})"
                )

            warnings.append(
                f"[WARNING] 连接跨簇{len(cross_cluster_violations)}个，扣{connection_penalty:.1f}分：{'; '.join(violation_details)}"
            )

        # 多簇归属扣分（最多25分）
        if satellite_multi_cluster_violations:
            satellite_penalty = min(len(satellite_multi_cluster_violations) * 5, 12.5)
            isolation_penalty += satellite_penalty

            # 构建卫星多簇详情 - 显示全部
            sat_details = []
            for sat in satellite_multi_cluster_violations:
                clusters = satellite_to_clusters[sat]
                sat_details.append(f"卫星{sat}(簇{clusters})")

            warnings.append(
                f"[WARNING] 卫星多簇{len(satellite_multi_cluster_violations)}个，扣{satellite_penalty:.1f}分：{'; '.join(sat_details)}"
            )

        if target_multi_cluster_violations:
            target_penalty = min(len(target_multi_cluster_violations) * 5, 12.5)
            isolation_penalty += target_penalty

            # 构建目标多簇详情 - 显示全部
            target_details = []
            for target in target_multi_cluster_violations:
                clusters = target_to_clusters[target]
                target_details.append(f"目标{target}(簇{clusters})")

            warnings.append(
                f"[WARNING] 目标多簇{len(target_multi_cluster_violations)}个，扣{target_penalty:.1f}分：{'; '.join(target_details)}"
            )

        # 累加隔离性扣分到总扣分
        score_penalty += isolation_penalty

        # 确保扣分不超过100分
        score_penalty = min(score_penalty, 100)
        final_score = int(100 - score_penalty)

        # 构建格式化信息文本
        cross_cluster_rate = (
            len(cross_cluster_violations) / total_valid_connections
            if total_valid_connections > 0
            else 0
        )

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
        summary = (
            f"[SUMMARY] 覆盖率:{target_coverage_rate:.1%}, 跨簇率:{cross_cluster_rate:.1%}, "
            f"多簇卫星:{len(satellite_multi_cluster_violations)}个, 多簇目标:{len(target_multi_cluster_violations)}个, "
            f"得分:{final_score}/100"
        )
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

        # === 2. 构建当前和历史分簇的连接关系映射 ===
        # 构建历史分簇的连接关系：记录每个目标/卫星与哪些其他实体在同一簇
        last_target_companions = {}  # 目标 -> 与它同簇的卫星集合
        last_satellite_companions = {}  # 卫星 -> 与它同簇的其他卫星集合
        last_target_to_master = {}  # 目标 -> 它所在簇的主节点
        last_satellite_to_master = {}  # 卫星 -> 它所在簇的主节点

        for cluster in last_clusters:
            master = cluster.master
            sats = set(cluster.sats)
            targets = set(cluster.targets)
            
            # 记录目标的伙伴关系
            for target in targets:
                last_target_companions[target] = sats.copy()
                last_target_to_master[target] = master
            
            # 记录卫星的伙伴关系  
            for satellite in sats:
                last_satellite_companions[satellite] = sats - {satellite}  # 排除自己
                last_satellite_to_master[satellite] = master

        # 构建当前分簇的连接关系映射
        current_target_companions = {}
        current_satellite_companions = {}
        current_target_to_master = {}
        current_satellite_to_master = {}

        for cluster in current_clusters:
            master = cluster.master
            sats = set(cluster.sats)
            targets = set(cluster.targets)
            
            # 记录目标的伙伴关系
            for target in targets:
                current_target_companions[target] = sats.copy()
                current_target_to_master[target] = master
                
            # 记录卫星的伙伴关系
            for satellite in sats:
                current_satellite_companions[satellite] = sats - {satellite}  # 排除自己
                current_satellite_to_master[satellite] = master

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
        common_targets = set(last_target_companions.keys()) & set(
            current_target_companions.keys()
        )
        total_targets = len(
            set(last_target_companions.keys()) | set(current_target_companions.keys())
        )

        for target in common_targets:
            last_companions = last_target_companions[target]
            current_companions = current_target_companions[target]
            
            # 获取当前时间切片中实际能观测到这个目标的卫星
            visible_sats_for_target = target_visibility.get(target, set())
            
            # 筛选出在当前时间切片中仍然可见的历史伙伴卫星
            still_visible_last_companions = last_companions & visible_sats_for_target
            
            # 判断是否真正发生了切换：
            # 如果历史伙伴卫星中有些在当前仍可见，但却不在当前伙伴列表中，这是真正的切换
            lost_visible_companions = still_visible_last_companions - current_companions
            
            # 只有当存在"本来可见但被移除"的卫星时，才认为是不合理的切换
            if len(lost_visible_companions) > 0:
                # 这是真正的切换：有卫星本来可以观测目标，但被分配到其他簇了
                target_switches.append(
                    {
                        "target": target,
                        "last_companions": last_companions,
                        "current_companions": current_companions,
                        "lost_visible_companions": lost_visible_companions,
                    }
                )
            else:
                # 记录簇扩展/收缩情况（不扣分，只是信息记录）
                added_sats = current_companions - last_companions
                removed_sats = last_companions - current_companions
                
                # 区分合理移除和新增
                if removed_sats:
                    # 检查移除的卫星是否还能观测目标
                    still_visible_removed = removed_sats & visible_sats_for_target
                    invisible_removed = removed_sats - visible_sats_for_target
                    
                    # 如果移除的卫星中有仍然可见的，这也是不合理的切换
                    if still_visible_removed:
                        # 这些卫星本来可以观测目标，却被移除了，也算切换
                        target_switches.append(
                            {
                                "target": target,
                                "last_companions": last_companions,
                                "current_companions": current_companions,
                                "lost_visible_companions": still_visible_removed,
                            }
                        )
                    elif invisible_removed and added_sats:
                        # 有不可见的卫星被移除，有新卫星加入
                        adjustments = []
                        if added_sats:
                            adjustments.append(f"增加卫星{added_sats}")
                        if invisible_removed:
                            adjustments.append(f"移除不可见卫星{invisible_removed}")
                        warnings.append(
                            f"[INFO] 目标{target}簇调整：{', '.join(adjustments)}"
                        )
                    elif not still_visible_removed and added_sats:
                        # 只是移除了不可见的卫星，加入了新卫星
                        adjustments = []
                        if added_sats:
                            adjustments.append(f"增加卫星{added_sats}")
                        if removed_sats:
                            adjustments.append(f"移除不可见卫星{removed_sats}")
                        warnings.append(
                            f"[INFO] 目标{target}簇调整：{', '.join(adjustments)}"
                        )
                elif added_sats:
                    # 只是增加了卫星
                    warnings.append(
                        f"[INFO] 目标{target}簇调整：增加卫星{added_sats}"
                    )

        target_switch_count = len(target_switches)
        target_switch_rate = (
            target_switch_count / total_targets if total_targets > 0 else 0
        )
        
        # 计算总的丢失卫星数量
        total_lost_satellites = sum(
            len(switch.get('lost_visible_companions', set())) 
            for switch in target_switches
        )

        # 目标切换惩罚 - 基于丢失的卫星数量
        if target_switch_count > 0:  # 任何目标切换都要扣分
            # 主要基于丢失卫星数量计算惩罚，目标切换率为次要因素
            satellite_loss_penalty = total_lost_satellites * 5  # 每丢失一个卫星扣5分
            rate_penalty = target_switch_rate * 20  # 切换率最多贡献20分
            target_penalty = min(satellite_loss_penalty + rate_penalty, 50)  # 总共最多50分
            
            score_penalty += target_penalty

            # 构建目标切换详情 - 显示丢失的可见伙伴卫星
            switch_details = []
            for switch in target_switches:
                lost_sats = switch.get('lost_visible_companions', set())
                switch_details.append(
                    f"目标{switch['target']}(丢失可见伙伴{lost_sats})"
                )

            warnings.append(
                f"[WARNING] 目标切换{target_switch_count}个，丢失{total_lost_satellites}个可见伙伴，扣{target_penalty:.1f}分：{'; '.join(switch_details)}"
            )

        # === 4. 卫星稳定性验证和扣分（最多30分）===
        # 首先收集已经因为目标切换而被惩罚的卫星
        satellites_penalized_for_targets = set()
        for switch in target_switches:
            lost_sats = switch.get('lost_visible_companions', set())
            satellites_penalized_for_targets.update(lost_sats)
        
        satellite_switches = []
        justified_satellite_switches = 0
        common_satellites = set(last_satellite_companions.keys()) & set(
            current_satellite_companions.keys()
        )
        total_satellites = len(
            set(last_satellite_companions.keys())
            | set(current_satellite_companions.keys())
        )

        for satellite in common_satellites:
            last_companions = last_satellite_companions[satellite]
            current_companions = current_satellite_companions[satellite]
            
            # 判断是否真正发生了切换：只有当伙伴卫星集合完全没有交集时才认为是切换
            intersection = last_companions & current_companions
                
            # 只有当两个集合完全没有交集时，才认为是真正的切换
            if len(intersection) == 0 and len(last_companions) > 0 and len(current_companions) > 0:
                # 检查豁免条件：卫星是否在原簇已成孤岛
                connected_sats = sat_connectivity.get(satellite, set())

                # 如果卫星在当前时间切片，与旧伙伴卫星已无任何连接
                if not last_companions.intersection(connected_sats):
                    justified_satellite_switches += 1
                    warnings.append(
                        f"[INFO] 卫星{satellite}切换被豁免：与原伙伴卫星{last_companions}已无连接"
                    )
                    continue  # 跳过，不计入惩罚

                # 检查这个卫星是否已经因为目标切换被惩罚过了
                if satellite in satellites_penalized_for_targets:
                    warnings.append(
                        f"[INFO] 卫星{satellite}切换不重复扣分：已在目标切换中惩罚"
                    )
                    continue  # 跳过，避免重复惩罚

                satellite_switches.append(
                    {
                        "satellite": satellite,
                        "last_companions": last_companions,
                        "current_companions": current_companions,
                    }
                )
            else:
                # 记录簇扩展/收缩情况（不扣分，只是信息记录）
                if len(intersection) > 0:
                    added_sats = current_companions - last_companions
                    removed_sats = last_companions - current_companions
                    if added_sats or removed_sats:
                        # 构建调整描述，只显示非空的集合
                        adjustments = []
                        if added_sats:
                            adjustments.append(f"增加伙伴{added_sats}")
                        if removed_sats:
                            adjustments.append(f"移除伙伴{removed_sats}")
                        warnings.append(
                            f"[INFO] 卫星{satellite}簇调整：{', '.join(adjustments)}"
                        )

        satellite_switch_count = len(satellite_switches)
        satellite_switch_rate = (
            satellite_switch_count / total_satellites if total_satellites > 0 else 0
        )

        # 卫星切换惩罚
        if satellite_switch_count > 0:  # 任何卫星切换都要扣分
            satellite_penalty = min(
                satellite_switch_rate * 100, 30
            )  # 按比例扣分，最多30分
            score_penalty += satellite_penalty

            # 构建卫星切换详情 - 显示伙伴卫星集合变化
            switch_details = []
            for switch in satellite_switches:
                switch_details.append(
                    f"卫星{switch['satellite']}(伙伴{switch['last_companions']}→{switch['current_companions']})"
                )

            warnings.append(
                f"[WARNING] 卫星切换{satellite_switch_count}个，扣{satellite_penalty:.1f}分：{'; '.join(switch_details)}"
            )

        # === 5. Jaccard相似度验证和扣分（最多20分）===
        jaccard_similarities = []
        max_clusters = max(len(last_clusters), len(current_clusters))

        for i in range(max_clusters):
            last_cluster_items = set()
            current_cluster_items = set()

            if i < len(last_clusters):
                last_cluster_items = set(
                    last_clusters[i].targets + last_clusters[i].sats
                )
            if i < len(current_clusters):
                current_cluster_items = set(
                    current_clusters[i].targets + current_clusters[i].sats
                )

            if last_cluster_items or current_cluster_items:
                intersection = len(last_cluster_items & current_cluster_items)
                union = len(last_cluster_items | current_cluster_items)
                jaccard = intersection / union if union > 0 else 0
                jaccard_similarities.append(jaccard)

        avg_jaccard = (
            sum(jaccard_similarities) / len(jaccard_similarities)
            if jaccard_similarities
            else 0
        )

        # Jaccard相似度惩罚
        jaccard_penalty = 0
        if avg_jaccard < 0.8:
            unjustified_switches = target_switch_count + satellite_switch_count
            total_switches = (
                unjustified_switches
                + justified_target_switches
                + justified_satellite_switches
            )

            if unjustified_switches > 0 and total_switches > 0:
                # 计算潜在的Jaccard惩罚
                potential_jaccard_penalty = (1 - (avg_jaccard / 0.8)) * 20

                # 根据不合理切换的比例来缩放惩罚
                unjustified_ratio = unjustified_switches / total_switches
                jaccard_penalty = potential_jaccard_penalty * unjustified_ratio

                warnings.append(
                    f"[WARNING] 簇重叠率{avg_jaccard:.1%}低于80%阈值，因不合理切换占比{unjustified_ratio:.1%}，扣{jaccard_penalty:.1f}分"
                )
            elif unjustified_switches == 0:
                warnings.append(
                    f"[INFO] Jaccard相似度惩罚被豁免：所有成员切换均为合理调整，虽然簇重叠率({avg_jaccard:.1%})较低，但不扣分。"
                )

        score_penalty += jaccard_penalty

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
        summary = (
            f"[SUMMARY] 目标切换率:{target_switch_rate:.1%}({target_switch_count}/{total_targets}, 豁免{justified_target_switches}个), "
            f"卫星切换率:{satellite_switch_rate:.1%}({satellite_switch_count}/{total_satellites}, 豁免{justified_satellite_switches}个), "
            f"簇重叠率:{avg_jaccard:.1%}, 得分:{final_score}/100"
        )
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

        通信代价计算说明：
        - 星座总代价：找到连通度最高的卫星作为整个星座的最优主节点，计算所有其他卫星到此主节点的通信代价之和
        - 路径规划：使用sat_edges提供的连接关系作为跳板进行最短路径计算
        - 分簇代价：簇内同步代价 + 簇间主节点通信代价
        - 效率评估：分簇方案的总代价 vs 星座最优主节点方案的代价

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
        constellation_satellites = set()  # 从sat_attrs中获取的所有卫星

        for sat_attr in conversation.input.sat_attrs:
            sat_positions[sat_attr.id] = sat_attr.pos
            constellation_satellites.add(sat_attr.id)

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
                invalid_master_clusters.append(
                    {
                        "cluster_id": cluster_idx,
                        "master": master_sat,
                        "sats": cluster_sats,
                    }
                )
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
                isolated_satellite_details.append(
                    {
                        "cluster_id": cluster_idx,
                        "master": master_sat,
                        "isolated_sats": isolated_satellites,
                    }
                )

        # 如果存在致命错误，直接扣满分
        if invalid_master_clusters or isolated_satellite_details:
            error_details = []

            if invalid_master_clusters:
                master_errors = []
                for cluster_info in invalid_master_clusters:
                    master_errors.append(
                        f"簇{cluster_info['cluster_id']}主节点{cluster_info['master']}不在簇内"
                    )
                error_details.append(f"主节点无效: {'; '.join(master_errors)}")

            if isolated_satellite_details:
                isolated_errors = []
                for detail in isolated_satellite_details:
                    isolated_sats_str = ", ".join(map(str, detail["isolated_sats"]))
                    isolated_errors.append(
                        f"簇{detail['cluster_id']}内卫星{isolated_sats_str}无法连通主节点{detail['master']}"
                    )
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

        # 计算簇内同步代价 TODO 计算通信代价应该考虑和主节点的健康度相关
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
            cluster_cost_details.append(
                {
                    "cluster_id": cluster_idx,
                    "intra_cost": intra_cluster_cost,
                    "master": master_sat,
                    "size": len(cluster_sats),
                }
            )

        # 计算全网同步代价（主节点间通信）
        masters = [cluster.master for cluster in conversation.response.clusters]

        for i, master1 in enumerate(masters):
            for master2 in masters[i + 1 :]:
                path_cost = self._find_shortest_path_cost(
                    master1, master2, sat_distances, constellation_satellites
                )

                if path_cost is not None:
                    total_inter_cluster_cost += path_cost
                # 如果无法连通，则不计算代价（忽略此连接）

        total_cost = total_intra_cluster_cost + total_inter_cluster_cost

        # 计算星座总通信代价作为基准（找到连通度最高的卫星作为主节点，计算所有卫星到主节点的通信代价）
        constellation_satellites = set()
        for sat_attr in conversation.input.sat_attrs:
            constellation_satellites.add(sat_attr.id)
        
        # 找到连通度最高的卫星作为整个星座的最优主节点
        best_constellation_master, total_constellation_cost, constellation_connectivity = self._find_best_constellation_master(
            constellation_satellites, sat_distances
        )
        
        if best_constellation_master is not None:
            # 已经在_find_best_constellation_master中计算过总代价，直接使用
            warnings.append(f"[INFO] 星座最优主节点: 卫星{best_constellation_master}, 总代价: {total_constellation_cost:.1f}km, 连通度: {constellation_connectivity}")
        else:
            # 如果无法找到合适的主节点，返回失败
            warnings.append("[WARNING] 无法找到星座最优主节点")
            total_constellation_cost = float('inf')

        # 通信效率评估（按比例扣分）
        cost_penalty = 0
        if total_constellation_cost > 0:
            cost_ratio = total_cost / total_constellation_cost
            efficiency_improvement = (1 - cost_ratio) * 100  # 效率提升百分比

            # 按代价比例扣分：代价比例越高，扣分越多
            if cost_ratio > 0.1:  # 超过10%的星座最优代价开始扣分
                # 线性扣分：10%时扣0分，100%时扣满100分
                cost_penalty = min((cost_ratio - 0.1) / 0.9 * 100, 100)
                warnings.append(
                    f"[WARNING] 分了{len(conversation.response.clusters)}个簇后通信代价仍然很高，扣{cost_penalty:.1f}分：当前总通信距离{total_cost:.1f}km，是星座最优主节点方案通信距离的{cost_ratio:.1%}，只节省了{efficiency_improvement:.1f}%的通信成本"
                )
            else:
                warnings.append(
                    f"[INFO] 分了{len(conversation.response.clusters)}个簇后通信效率很好，不扣分：当前总通信距离{total_cost:.1f}km，仅占星座最优主节点方案通信距离的{cost_ratio:.1%}，节省了{efficiency_improvement:.1f}%的通信成本"
                )
        else:
            warnings.append("[WARNING] 无法计算星座最优基准代价或当前不构成观测星座，跳过效率评估")

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
        avg_cluster_cost = (
            total_intra_cluster_cost / len(conversation.response.clusters)
            if conversation.response.clusters
            else 0
        )
        cost_ratio = (
            total_cost / total_constellation_cost if total_constellation_cost > 0 else 0
        )

        summary = (
            f"[SUMMARY] 总代价:{total_cost:.1f}km, 簇内:{total_intra_cluster_cost:.1f}km, "
            f"分簇后簇间:{total_inter_cluster_cost:.1f}km, 平均簇内:{avg_cluster_cost:.1f}km, "
            f"vs星座最优:{total_constellation_cost:.1f}km, 代价比例:{cost_ratio:.1%}, 得分:{final_score}/100"
        )
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
        1. 致命错误检测：目标被分配到簇内，但簇内无卫星可观测 → 扣全部100分
        2. 观测重数评估：基于簇内目标的平均观测重数进行评分。
           - 平均观测重数越高，得分越高。
           - 1.0为及格线，2.5为理想目标（满分）。

        Args:
            conversation: 单个对话数据

        Returns:
            ValidationDetail: 验证详情对象
        """
        # 初始化
        errors = []
        warnings = []
        cluster_stats = []

        # 构建目标到观测卫星的映射
        target_to_sats = defaultdict(set)
        for edge in conversation.input.target_edges:
            target_to_sats[edge.target_id].add(edge.sat_id)

        # === 1. 逐簇分析观测重数 ===
        unobserved_target_errors = []
        for cluster_idx, cluster in enumerate(conversation.response.clusters):
            cluster_sats = set(cluster.sats)
            cluster_targets = set(cluster.targets)

            if not cluster_targets:
                warnings.append(f"[INFO] 簇 {cluster_idx} 无目标，跳过观测效能评估。")
                continue

            observation_counts = defaultdict(int)
            total_observation_multiplicity = 0

            for target in cluster_targets:
                # 计算能观测到此目标的、且在当前簇内的卫星数量
                observing_sats_in_cluster = target_to_sats[target] & cluster_sats
                count = len(observing_sats_in_cluster)
                observation_counts[count] += 1
                total_observation_multiplicity += count

            # 检查致命错误：目标在簇内但无法被观测
            if observation_counts[0] > 0:
                unobserved_targets = [
                    target
                    for target in cluster_targets
                    if len(target_to_sats[target] & cluster_sats) == 0
                ]
                unobserved_target_count = len(unobserved_targets)
                unobserved_targets_str = ", ".join(map(str, unobserved_targets))
                unobserved_target_errors.append(
                    f"簇{cluster_idx}中有{unobserved_target_count}个目标({unobserved_targets_str})无法被簇内任何卫星观测"
                )

            # 计算当前簇的平均观测重数
            avg_multiplicity = (
                total_observation_multiplicity / len(cluster_targets)
                if cluster_targets
                else 0
            )

            cluster_stats.append(
                {
                    "cluster_id": cluster_idx,
                    "avg_multiplicity": avg_multiplicity,
                    "distribution": dict(sorted(observation_counts.items())),
                }
            )

        # === 2. 致命错误处理 ===
        if unobserved_target_errors:
            error_msg = f"[ERROR] 存在目标分配错误，观测效能为0分: {'; '.join(unobserved_target_errors)}"
            return ValidationDetail(
                validation_type="observation_efficiency_validation",
                score=0,
                info=error_msg,
            )

        # === 3. 计算总体得分 ===
        if not cluster_stats:
            return ValidationDetail(
                validation_type="observation_efficiency_validation",
                score=100,
                info="[INFO] 无需评估的簇，默认满分",
            )

        # 计算全局平均观测重数
        total_avg_multiplicity = sum(s["avg_multiplicity"] for s in cluster_stats)
        global_avg_multiplicity = total_avg_multiplicity / len(cluster_stats)

        # 评分标准：
        # - 平均观测重数 >= 2.5: 满分 100
        # - 平均观测重数 <= 1.0: 0分
        # - 在 (1.0, 2.5) 区间内线性插值
        min_threshold = 1.0
        ideal_threshold = 2.5

        if global_avg_multiplicity >= ideal_threshold:
            final_score = 100
        elif global_avg_multiplicity <= min_threshold:
            final_score = 0
        else:
            final_score = int(
                (
                    (global_avg_multiplicity - min_threshold)
                    / (ideal_threshold - min_threshold)
                )
                * 100
            )

        # === 4. 构建详细信息 ===
        info_parts = []
        
        # 如果有扣分，添加警告说明
        if final_score < 100:
            score_loss = 100 - final_score
            if global_avg_multiplicity < min_threshold:
                warnings.append(f"[WARNING] 观测重数过低，扣{score_loss}分：全局平均观测重数{global_avg_multiplicity:.2f}低于最低阈值{min_threshold}")
            elif global_avg_multiplicity < ideal_threshold:
                warnings.append(f"[WARNING] 观测重数不达标，扣{score_loss}分：全局平均观测重数{global_avg_multiplicity:.2f}低于理想阈值{ideal_threshold}")
        
        if warnings:
            info_parts.extend(warnings)

        # 簇详情
        cluster_details_str = []
        for stat in cluster_stats:
            dist_str = ", ".join(
                [f"{k}重x{v}" for k, v in stat["distribution"].items()]
            )
            cluster_details_str.append(
                f"簇{stat['cluster_id']}(均重:{stat['avg_multiplicity']:.2f}, 分布:[{dist_str}])"
            )

        # 摘要信息
        summary = (
            f"[SUMMARY] 全局平均观测重数:{global_avg_multiplicity:.2f}, "
            f"得分:{final_score}/100 | 簇详情: {'; '.join(cluster_details_str)}"
        )
        info_parts.append(summary)

        info_text = "\n".join(info_parts)

        return ValidationDetail(
            validation_type="observation_efficiency_validation",
            score=final_score,
            info=info_text,
        )

    def _find_shortest_path_cost(
        self, start_sat, target_sat, sat_distances: Dict, cluster_sats
    ):
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
        distances = {sat: float("inf") for sat in cluster_sats}
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

    def _find_best_constellation_master(
        self, constellation_satellites, sat_distances: Dict
    ):
        """找到整个星座中连通度最高的卫星作为最优主节点

        通过计算每个卫星能够连通到的其他卫星数量（包括多跳连接），
        选择连通度最高的卫星作为主节点，同时计算总通信代价。

        Args:
            constellation_satellites: 星座中所有卫星的集合
            sat_distances: 卫星间距离映射

        Returns:
            tuple: (最优主节点的ID, 对应的总通信代价, 连通度数量)，如果无法找到则返回(None, 0, 0)
        """
        if not constellation_satellites:
            return None, 0, 0

        best_master = None
        best_total_cost = 0
        max_reachable_count = -1
        total_satellites = len(constellation_satellites) - 1  # 排除自己

        # 尝试每个卫星作为主节点，选择连通度最高的
        for candidate_master in constellation_satellites:
            total_cost = 0
            reachable_count = 0

            # 计算其他所有卫星到这个候选主节点的通信代价
            for satellite in constellation_satellites:
                if satellite == candidate_master:
                    continue

                path_cost = self._find_shortest_path_cost(
                    satellite, candidate_master, sat_distances, constellation_satellites
                )

                if path_cost is not None:
                    total_cost += path_cost
                    reachable_count += 1

            # 选择连通度最高的卫星作为主节点
            # 如果连通度相同，选择通信代价最小的
            if reachable_count > max_reachable_count or (
                reachable_count == max_reachable_count and 
                reachable_count > 0 and 
                (best_master is None or total_cost < best_total_cost)
            ):
                max_reachable_count = reachable_count
                best_total_cost = total_cost
                best_master = candidate_master

        return best_master, best_total_cost, max_reachable_count

    def evaluate_all_results(
        self, validation_results: List[ValidationItem]
    ) -> Dict[str, Any]:
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
                "perfect_score_rate": 0,
            }

        # 定义权重（用于显示和统计，实际计算在ValidationItem.score中）
        weights = {
            "correctness_validation": 0.4,  # 正确性和隔离性 40%
            "stability_validation": 0.3,  # 分簇稳定性 30%
            "communication_cost_validation": 0.15,  # 通信代价 15%
            "observation_efficiency_validation": 0.15,  # 观测效能 15%
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
            "0-9": 0,
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
            "correctness_validation": {
                "scores": [],
                "max_score": 100,
                "avg_score": 0,
                "loss_count": 0,
                "loss_reasons": [],
            },
            "stability_validation": {
                "scores": [],
                "max_score": 100,
                "avg_score": 0,
                "loss_count": 0,
                "loss_reasons": [],
            },
            "communication_cost_validation": {
                "scores": [],
                "max_score": 100,
                "avg_score": 0,
                "loss_count": 0,
                "loss_reasons": [],
            },
            "observation_efficiency_validation": {
                "scores": [],
                "max_score": 100,
                "avg_score": 0,
                "loss_count": 0,
                "loss_reasons": [],
            },
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
                    
                    # 提取所有ERROR
                    if "[ERROR]" in info:
                        error_parts = info.split("[ERROR]")[1:]
                        for error_part in error_parts:
                            # 找到第一个换行符或下一个标签的位置
                            if "\n" in error_part:
                                error_content = error_part.split("\n")[0].strip()
                            else:
                                error_content = error_part.strip()
                            if error_content:
                                reasons.append(f"[ERROR] {error_content}")
                    
                    # 提取所有WARNING
                    if "[WARNING]" in info:
                        warning_parts = info.split("[WARNING]")[1:]
                        for warning_part in warning_parts:
                            # 找到第一个换行符或下一个标签的位置
                            if "\n" in warning_part:
                                warning_content = warning_part.split("\n")[0].strip()
                            else:
                                warning_content = warning_part.strip()
                            if warning_content:
                                reasons.append(f"[WARNING] {warning_content}")

                    loss_reason = {
                        "sample_index": item_idx,
                        "timestamp": item.input.timestamp,  # 添加时间戳信息
                        "validation_type": validation_type,
                        "score": score,
                        "max_score": max_possible,
                        "loss_points": loss_points,
                        "reasons": reasons if reasons else ["未知原因"],
                        "full_info": info,
                    }
                    loss_item_reasons.append(loss_reason)
                    validation_type_stats[validation_type]["loss_reasons"].append(
                        loss_reason
                    )

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
                "most_common_loss_type": self._get_most_common_loss_type(
                    loss_item_reasons
                ),
            },
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
        print("   • 通信代价: 100分 × 15% = 15分")
        print("   • 观测效能: 100分 × 15% = 15分")
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
        }

        for validation_type, stats in evaluation_result[
            "validation_type_stats"
        ].items():
            type_name = type_names.get(validation_type, validation_type)
            print(f"   {type_name}:")
            print(f"     平均分: {stats['avg_score']:.1f}/{stats['max_score']}")
            print(f"     分数范围: {stats['min_score']}-{stats['max_score_achieved']}")
            print(
                f"     丢分率: {stats['loss_rate']:.1%} ({stats['loss_count']}个样本)"
            )

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

                for reason, count in sorted(
                    reason_counts.items(), key=lambda x: x[1], reverse=True
                ):
                    print(f"     - {reason}: {count}次")
        else:
            print("   🎉 没有丢分项目！")

        print("=" * 80)


if __name__ == "__main__":
    import sys
    timestamp = get_current_timestamp()
    data_path = get_data_dir() / "cluster_results_sharegpt_training_data/max_overlap_alg_for_raw_constellation_data_scenario_1_with_history.jsonl"
    raw_data: List[LLMConversationMessage] = load_sharegpt_data(data_path)
    validator = ClusterDataValidator()

    print("🚀 开始验证卫星分簇结果...")
    validation_results: List[ValidationItem] = validator.validate_output(raw_data)

    print("📊 生成详细统计报告...")
    evaluation_result = validator.evaluate_all_results(validation_results)

    # 打印统计摘要
    validator.print_evaluation_summary(evaluation_result)

    # 保存验证结果
    output_file = (
        get_data_dir()
        / f"cluster_results_sharegpt_training_data/{data_path.stem}_validation_result_{timestamp}.jsonl"
    )
    with open(output_file, "w", encoding="utf-8") as f:
        for item in validation_results:
            f.write(
                json.dumps(item, default=lambda o: o.__dict__, ensure_ascii=False)
                + "\n"
            )

    # 保存统计结果
    stats_file = get_data_dir() / f"cluster_results_sharegpt_training_data/{data_path.stem}_validation_stats_{timestamp}.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_result, f, ensure_ascii=False, indent=2)

    print(f"✅ 验证完成！结果已保存到:")
    print(f"   验证结果: {output_file}")
    print(f"   统计报告: {stats_file}")
