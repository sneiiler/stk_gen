#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClusterDataValidator_dist.py

卫星集群配置验证框架分发版本

本模块为卫星集群配置提供全面的验证功能，包括正确性验证、稳定性分析、
通信成本评估和观测效率评估。

Author: Kaifeng
File: ClusterDataValidator_dist.py
Version: 1.0.0
Created: 2025-07-24


Usage:
    from ClusterDataValidator_dist import ClusterDataValidator
    
    validator = ClusterDataValidator()
    results = validator.validate_output(conversation_data)
"""

import json
import heapq
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from collections import defaultdict
from pydantic import BaseModel, Field
from tqdm import tqdm


class SatelliteAttributes(BaseModel):
    id: str | int = Field(..., description="卫星ID")
    health: float = Field(..., ge=0, le=10, description="卫星健康状态 (0-10)")
    pos: List[float] = Field(..., description="卫星ECEF位置坐标 [x, y, z] km")


class SatelliteEdge(BaseModel):
    from_sat: str | int = Field(..., description="起始卫星ID")
    to_sat: str | int = Field(..., description="目标卫星ID")
    distance: float = Field(..., description="卫星距离，单位km")


class TargetEdge(BaseModel):
    sat_id: str | int = Field(..., description="起始卫星ID")
    target_id: str | int = Field(..., description="目标ID")
    quality: float = Field(..., description="连接质量 (0-1)")


class ClusterInfo(BaseModel):
    timestamp: Optional[str] = Field(..., description="ISO8601格式的时间戳字符串")
    cluster_id: str | int = Field(description="分簇ID")
    master: str | int = Field(description="主节点卫星ID")
    sats: List[str | int] = Field(description="分簇中的卫星ID列表")
    targets: List[str | int] = Field(description="分簇观测的目标ID列表")


class RawConstellationDataModel(BaseModel):
    timestamp: str = Field(..., description="ISO8601格式的时间戳字符串")
    sat_attrs: List[SatelliteAttributes] = Field(..., description="卫星属性列表")
    sat_edges: List[SatelliteEdge] = Field(..., description="卫星间连接关系列表")
    target_edges: List[TargetEdge] = Field(..., description="卫星到目标的连接关系列表")
    history_cluster_result: Optional[List[List[ClusterInfo]]] = Field(
        ..., description="上n次分簇结果"
    )


class SatelliteClusterOutput(BaseModel):
    chain_of_thought: Optional[str] = Field(
        description="推理过程，大模型生成阶段不需要填写，后期封装"
    )
    clusters: List[ClusterInfo] = Field(description="划分的卫星分簇列表")

    def to_think_json(self):
        thought_content = self.chain_of_thought or ""
        return (
            "<think>"
            + thought_content
            + "</think>"
            + json.dumps(
                [cluster.model_dump() for cluster in self.clusters],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )


class LLMConversationMessage(BaseModel):
    instruction: str = Field(..., description="给模型的指令，通常是系统提示词")
    input: RawConstellationDataModel = Field(..., description="输入的卫星星座数据")
    response: SatelliteClusterOutput = Field(
        ..., description="包含推理过程和分簇结果的完整响应"
    )


class ValidationDetail(BaseModel):
    validation_type: Literal[
        "correctness_validation",
        "stability_validation", 
        "communication_cost_validation",
        "observation_efficiency_validation",
    ] = Field(..., description="验证类型")
    score: int = Field(..., description="分项得分")
    info: str = Field(..., description="警告信息")


class ValidationItem(BaseModel):
    input: RawConstellationDataModel = Field(..., description="输入数据")
    response: List[ClusterInfo] = Field(..., description="模型响应提取的数据")
    validation_details: List[ValidationDetail] = Field(..., description="验证详情")

    @property
    def score(self) -> float:
        weights = {
            "correctness_validation": 0.4,
            "stability_validation": 0.3,
            "communication_cost_validation": 0.15,
            "observation_efficiency_validation": 0.15,
        }
        weighted_score = 0.0
        for detail in self.validation_details:
            weight = weights.get(detail.validation_type, 0)
            weighted_score += detail.score * weight
        return weighted_score


class ClusterDataValidator:
    def __init__(self):
        pass

    def validate_output(
        self, input_data: List[LLMConversationMessage]
    ) -> List[ValidationItem]:
        validation_results = []
        try:
            for conversation in tqdm(input_data, desc="Processing"):
                validation_details = []
                
                correctness_detail = self._validate_correctness_and_isolation_for_single_slice(conversation)
                validation_details.append(correctness_detail)
                
                stability_detail = self._validate_stability_for_single_slice(conversation)
                validation_details.append(stability_detail)
                
                cost_detail = self._validate_communication_cost_for_single_slice(conversation)
                validation_details.append(cost_detail)
                
                efficiency_detail = self._validate_observation_efficiency_for_single_slice(conversation)
                validation_details.append(efficiency_detail)
                
                validation_item = ValidationItem(
                    input=conversation.input,
                    response=conversation.response.clusters,
                    validation_details=validation_details,
                )
                validation_results.append(validation_item)
        except Exception as e:
            print(f"Error: {e}")
        return validation_results

    def _validate_correctness_and_isolation_for_single_slice(
        self, conversation: LLMConversationMessage,
    ) -> ValidationDetail:
        valid_sat_target_connections = set()
        input_targets = set()
        input_satellites = set()

        for edge in conversation.input.target_edges:
            valid_sat_target_connections.add((edge.sat_id, edge.target_id))
            input_targets.add(edge.target_id)

        for edge in conversation.input.sat_edges:
            input_satellites.add(edge.from_sat)
            input_satellites.add(edge.to_sat)

        output_targets = set()
        output_satellites = set()
        target_cluster_map = {}
        satellite_cluster_map = {}

        for cluster_idx, cluster in enumerate(conversation.response.clusters):
            cluster_targets = set(cluster.targets)
            cluster_satellites = set(cluster.sats)

            for target in cluster_targets:
                target_cluster_map[target] = cluster_idx
                output_targets.add(target)

            for satellite in cluster_satellites:
                satellite_cluster_map[satellite] = cluster_idx
                output_satellites.add(satellite)

        invalid_targets = output_targets - input_targets
        invalid_satellites = output_satellites - input_satellites

        if invalid_targets or invalid_satellites:
            error_details = []
            if invalid_targets:
                target_list = sorted(list(invalid_targets))
                error_details.append(f"Invalid targets: {', '.join(map(str, target_list))}")
            if invalid_satellites:
                sat_list = sorted(list(invalid_satellites))
                error_details.append(f"Invalid satellites: {', '.join(map(str, sat_list))}")
            error_msg = f"[ERROR] Invalid elements: {'; '.join(error_details)}"
            return ValidationDetail(
                validation_type="correctness_validation",
                score=0,
                info=error_msg,
            )

        missing_targets = input_targets - output_targets
        if missing_targets:
            missing_list = sorted(list(missing_targets))
            error_msg = f"[ERROR] Missing targets: {', '.join(map(str, missing_list))}"
            return ValidationDetail(
                validation_type="correctness_validation",
                score=0,
                info=error_msg,
            )

        target_coverage_rate = 1.0
        score_penalty = 0
        warnings = []

        cross_cluster_violations = []
        total_valid_connections = 0

        for sat, target in valid_sat_target_connections:
            sat_cluster = satellite_cluster_map.get(sat)
            target_cluster = target_cluster_map.get(target)
            if sat_cluster is not None and target_cluster is not None:
                total_valid_connections += 1
                if sat_cluster != target_cluster:
                    cross_cluster_violations.append({"satellite": sat, "target": target})

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
            target for target, clusters in target_to_clusters.items() if len(clusters) > 1
        ]

        isolation_penalty = 0

        if cross_cluster_violations and total_valid_connections > 0:
            cross_cluster_rate = len(cross_cluster_violations) / total_valid_connections
            connection_penalty = cross_cluster_rate * 25
            isolation_penalty += connection_penalty
            warnings.append(f"[WARNING] Cross-cluster connections: {len(cross_cluster_violations)}, penalty: {connection_penalty:.1f}")

        if satellite_multi_cluster_violations:
            satellite_penalty = min(len(satellite_multi_cluster_violations) * 5, 12.5)
            isolation_penalty += satellite_penalty
            warnings.append(f"[WARNING] Multi-cluster satellites: {len(satellite_multi_cluster_violations)}, penalty: {satellite_penalty:.1f}")

        if target_multi_cluster_violations:
            target_penalty = min(len(target_multi_cluster_violations) * 5, 12.5)
            isolation_penalty += target_penalty
            warnings.append(f"[WARNING] Multi-cluster targets: {len(target_multi_cluster_violations)}, penalty: {target_penalty:.1f}")

        score_penalty += isolation_penalty
        score_penalty = min(score_penalty, 100)
        final_score = int(100 - score_penalty)

        cross_cluster_rate = (
            len(cross_cluster_violations) / total_valid_connections
            if total_valid_connections > 0
            else 0
        )

        info_parts = []
        if warnings:
            info_parts.extend(warnings)

        summary = (
            f"[SUMMARY] Coverage:{target_coverage_rate:.1%}, Cross-cluster rate:{cross_cluster_rate:.1%}, "
            f"Multi-cluster sats:{len(satellite_multi_cluster_violations)}, Multi-cluster targets:{len(target_multi_cluster_violations)}, "
            f"Score:{final_score}/100"
        )
        info_parts.append(summary)
        info_text = "\n".join(info_parts)

        return ValidationDetail(
            validation_type="correctness_validation",
            score=final_score,
            info=info_text,
        )

    def _validate_stability_for_single_slice(
        self, conversation: LLMConversationMessage,
    ) -> ValidationDetail:
        score_penalty = 0
        warnings = []
        info_logs = []

        history_results = conversation.input.history_cluster_result
        if not history_results or len(history_results) == 0:
            return ValidationDetail(
                validation_type="stability_validation",
                score=100,
                info="[INFO] No history data available",
            )

        last_clusters = history_results[-1] if history_results else []
        current_clusters = conversation.response.clusters

        if not last_clusters:
            return ValidationDetail(
                validation_type="stability_validation",
                score=0,
                info="[ERROR] Empty history data",
            )

        last_target_companions = {}
        last_satellite_companions = {}

        for cluster in last_clusters:
            sats = set(cluster.sats)
            targets = set(cluster.targets)
            
            for target in targets:
                last_target_companions[target] = sats.copy()
            
            for satellite in sats:
                last_satellite_companions[satellite] = sats - {satellite}

        current_target_companions = {}
        current_satellite_companions = {}

        for cluster in current_clusters:
            sats = set(cluster.sats)
            targets = set(cluster.targets)
            
            for target in targets:
                current_target_companions[target] = sats.copy()
                
            for satellite in sats:
                current_satellite_companions[satellite] = sats - {satellite}

        target_visibility = defaultdict(set)
        for edge in conversation.input.target_edges:
            target_visibility[edge.target_id].add(edge.sat_id)

        sat_connectivity = defaultdict(set)
        for edge in conversation.input.sat_edges:
            sat_connectivity[edge.from_sat].add(edge.to_sat)
            sat_connectivity[edge.to_sat].add(edge.from_sat)

        target_switches = []
        common_targets = set(last_target_companions.keys()) & set(current_target_companions.keys())
        total_targets = len(set(last_target_companions.keys()) | set(current_target_companions.keys()))

        for target in common_targets:
            last_companions = last_target_companions[target]
            current_companions = current_target_companions[target]
            
            visible_sats_for_target = target_visibility.get(target, set())
            still_visible_last_companions = last_companions & visible_sats_for_target
            lost_visible_companions = still_visible_last_companions - current_companions
            
            if len(lost_visible_companions) > 0:
                target_switches.append({
                    "target": target,
                    "lost_visible_companions": lost_visible_companions,
                })

        target_switch_count = len(target_switches)
        target_switch_rate = target_switch_count / total_targets if total_targets > 0 else 0
        
        total_lost_satellites = sum(
            len(switch.get('lost_visible_companions', set())) 
            for switch in target_switches
        )

        if target_switch_count > 0:
            satellite_loss_penalty = total_lost_satellites * 5
            rate_penalty = target_switch_rate * 20
            target_penalty = min(satellite_loss_penalty + rate_penalty, 50)
            score_penalty += target_penalty
            warnings.append(f"[WARNING] Target switches: {target_switch_count}, lost companions: {total_lost_satellites}, penalty: {target_penalty:.1f}")

        satellite_switches = []
        common_satellites = set(last_satellite_companions.keys()) & set(current_satellite_companions.keys())
        total_satellites = len(set(last_satellite_companions.keys()) | set(current_satellite_companions.keys()))

        for satellite in common_satellites:
            last_companions = last_satellite_companions[satellite]
            current_companions = current_satellite_companions[satellite]
            
            intersection = last_companions & current_companions
                
            if len(intersection) == 0 and len(last_companions) > 0 and len(current_companions) > 0:
                connected_sats = sat_connectivity.get(satellite, set())
                if not last_companions.intersection(connected_sats):
                    info_logs.append(f"[INFO] Satellite {satellite} switch exempted")
                    continue
                
                satellite_switches.append({
                    "satellite": satellite,
                    "last_companions": last_companions,
                    "current_companions": current_companions,
                })

        satellite_switch_count = len(satellite_switches)
        satellite_switch_rate = satellite_switch_count / total_satellites if total_satellites > 0 else 0

        if satellite_switch_count > 0:
            satellite_penalty = min(satellite_switch_rate * 100, 30)
            score_penalty += satellite_penalty
            warnings.append(f"[WARNING] Satellite switches: {satellite_switch_count}, penalty: {satellite_penalty:.1f}")

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

        jaccard_penalty = 0
        if avg_jaccard < 0.8:
            if target_switch_count + satellite_switch_count > 0:
                potential_jaccard_penalty = (1 - (avg_jaccard / 0.8)) * 20
                jaccard_penalty = potential_jaccard_penalty
                warnings.append(f"[WARNING] Low cluster overlap: {avg_jaccard:.1%}, penalty: {jaccard_penalty:.1f}")

        score_penalty += jaccard_penalty
        score_penalty = min(score_penalty, 100)
        final_score = int(100 - score_penalty)

        info_parts = []
        if info_logs:
            info_parts.extend(info_logs)
        if warnings:
            info_parts.extend(warnings)

        summary = (
            f"[SUMMARY] Target switch rate:{target_switch_rate:.1%}, "
            f"Satellite switch rate:{satellite_switch_rate:.1%}, "
            f"Cluster overlap:{avg_jaccard:.1%}, Score:{final_score}/100"
        )
        info_parts.append(summary)
        info_text = "\n".join(info_parts)

        return ValidationDetail(
            validation_type="stability_validation",
            score=final_score,
            info=info_text,
        )

    def _validate_communication_cost_for_single_slice(
        self, conversation: LLMConversationMessage,
    ) -> ValidationDetail:
        sat_positions = {}
        sat_distances = {}
        all_satellites = set()

        for sat_attr in conversation.input.sat_attrs:
            sat_positions[sat_attr.id] = sat_attr.pos
            all_satellites.add(sat_attr.id)

        for edge in conversation.input.sat_edges:
            sat_distances[(edge.from_sat, edge.to_sat)] = edge.distance
            sat_distances[(edge.to_sat, edge.from_sat)] = edge.distance

        score_penalty = 0
        warnings = []

        invalid_master_clusters = []
        isolated_satellite_details = []

        for cluster_idx, cluster in enumerate(conversation.response.clusters):
            cluster_sats = cluster.sats
            master_sat = cluster.master

            if master_sat not in cluster_sats:
                invalid_master_clusters.append({
                    "cluster_id": cluster_idx,
                    "master": master_sat,
                    "sats": cluster_sats,
                })
                continue

            isolated_satellites = []
            for member_sat in cluster_sats:
                if member_sat == master_sat:
                    continue

                path_cost = self._find_shortest_path_cost(
                    member_sat, master_sat, sat_distances, cluster_sats
                )

                if path_cost is None:
                    isolated_satellites.append(member_sat)

            if isolated_satellites:
                isolated_satellite_details.append({
                    "cluster_id": cluster_idx,
                    "master": master_sat,
                    "isolated_sats": isolated_satellites,
                })

        if invalid_master_clusters or isolated_satellite_details:
            error_details = []
            if invalid_master_clusters:
                error_details.append("Invalid masters")
            if isolated_satellite_details:
                error_details.append("Isolated satellites")
            error_msg = f"[ERROR] Communication issues: {'; '.join(error_details)}"
            return ValidationDetail(
                validation_type="communication_cost_validation",
                score=0,
                info=error_msg,
            )

        total_intra_cluster_cost = 0
        total_inter_cluster_cost = 0

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

        masters = [cluster.master for cluster in conversation.response.clusters]
        for i, master1 in enumerate(masters):
            for master2 in masters[i + 1 :]:
                path_cost = self._find_shortest_path_cost(
                    master1, master2, sat_distances, all_satellites
                )
                if path_cost is not None:
                    total_inter_cluster_cost += path_cost

        total_cost = total_intra_cluster_cost + total_inter_cluster_cost

        total_constellation_cost = 0
        for i, sat1 in enumerate(all_satellites):
            for sat2 in list(all_satellites)[i + 1 :]:
                path_cost = self._find_shortest_path_cost(
                    sat1, sat2, sat_distances, all_satellites
                )
                if path_cost is not None:
                    total_constellation_cost += path_cost

        cost_penalty = 0
        if total_constellation_cost > 0:
            cost_ratio = total_cost / total_constellation_cost
            if cost_ratio > 0.1:
                cost_penalty = min((cost_ratio - 0.1) / 0.9 * 100, 100)
                warnings.append(f"[WARNING] High communication cost, penalty: {cost_penalty:.1f}")

        score_penalty += cost_penalty
        score_penalty = min(score_penalty, 100)
        final_score = int(100 - score_penalty)

        info_parts = []
        if warnings:
            info_parts.extend(warnings)

        cost_ratio = total_cost / total_constellation_cost if total_constellation_cost > 0 else 0
        summary = (
            f"[SUMMARY] Total cost:{total_cost:.1f}km, "
            f"Cost ratio:{cost_ratio:.1%}, Score:{final_score}/100"
        )
        info_parts.append(summary)
        info_text = "\n".join(info_parts)

        return ValidationDetail(
            validation_type="communication_cost_validation",
            score=final_score,
            info=info_text,
        )

    def _validate_observation_efficiency_for_single_slice(
        self, conversation: LLMConversationMessage,
    ) -> ValidationDetail:
        warnings = []
        cluster_stats = []

        target_to_sats = defaultdict(set)
        for edge in conversation.input.target_edges:
            target_to_sats[edge.target_id].add(edge.sat_id)

        unobserved_target_errors = []
        for cluster_idx, cluster in enumerate(conversation.response.clusters):
            cluster_sats = set(cluster.sats)
            cluster_targets = set(cluster.targets)

            if not cluster_targets:
                continue

            observation_counts = defaultdict(int)
            total_observation_multiplicity = 0

            for target in cluster_targets:
                observing_sats_in_cluster = target_to_sats[target] & cluster_sats
                count = len(observing_sats_in_cluster)
                observation_counts[count] += 1
                total_observation_multiplicity += count

            if observation_counts[0] > 0:
                unobserved_targets = [
                    target for target in cluster_targets
                    if len(target_to_sats[target] & cluster_sats) == 0
                ]
                unobserved_target_errors.append(f"Cluster {cluster_idx}: {len(unobserved_targets)} unobserved targets")

            avg_multiplicity = (
                total_observation_multiplicity / len(cluster_targets)
                if cluster_targets
                else 0
            )

            cluster_stats.append({
                "cluster_id": cluster_idx,
                "avg_multiplicity": avg_multiplicity,
                "distribution": dict(sorted(observation_counts.items())),
            })

        if unobserved_target_errors:
            error_msg = f"[ERROR] Unobserved targets: {'; '.join(unobserved_target_errors)}"
            return ValidationDetail(
                validation_type="observation_efficiency_validation",
                score=0,
                info=error_msg,
            )

        if not cluster_stats:
            return ValidationDetail(
                validation_type="observation_efficiency_validation",
                score=100,
                info="[INFO] No clusters to evaluate",
            )

        total_avg_multiplicity = sum(s["avg_multiplicity"] for s in cluster_stats)
        global_avg_multiplicity = total_avg_multiplicity / len(cluster_stats)

        min_threshold = 1.0
        ideal_threshold = 2.5

        if global_avg_multiplicity >= ideal_threshold:
            final_score = 100
        elif global_avg_multiplicity <= min_threshold:
            final_score = 0
        else:
            final_score = int(
                ((global_avg_multiplicity - min_threshold) / (ideal_threshold - min_threshold)) * 100
            )

        info_parts = []
        
        if final_score < 100:
            score_loss = 100 - final_score
            if global_avg_multiplicity < min_threshold:
                warnings.append(f"[WARNING] Low observation multiplicity, penalty: {score_loss}")
            elif global_avg_multiplicity < ideal_threshold:
                warnings.append(f"[WARNING] Suboptimal observation multiplicity, penalty: {score_loss}")
        
        if warnings:
            info_parts.extend(warnings)

        summary = (
            f"[SUMMARY] Global avg multiplicity:{global_avg_multiplicity:.2f}, "
            f"Score:{final_score}/100"
        )
        info_parts.append(summary)
        info_text = "\n".join(info_parts)

        return ValidationDetail(
            validation_type="observation_efficiency_validation",
            score=final_score,
            info=info_text,
        )

    def _find_shortest_path_cost(self, start_sat, target_sat, sat_distances: Dict, cluster_sats):
        if start_sat == target_sat:
            return 0.0

        distances = {sat: float("inf") for sat in cluster_sats}
        distances[start_sat] = 0.0
        visited = set()
        priority_queue = [(0.0, start_sat)]

        while priority_queue:
            current_distance, current_sat = heapq.heappop(priority_queue)

            if current_sat in visited:
                continue

            visited.add(current_sat)

            if current_sat == target_sat:
                return current_distance

            for neighbor_sat in cluster_sats:
                if neighbor_sat in visited:
                    continue

                edge_distance = sat_distances.get((current_sat, neighbor_sat), None)
                if edge_distance is not None:
                    new_distance = current_distance + edge_distance

                    if new_distance < distances[neighbor_sat]:
                        distances[neighbor_sat] = new_distance
                        heapq.heappush(priority_queue, (new_distance, neighbor_sat))

        return None


if __name__ == "__main__":
    # Test data (second record from the original file)
    test_data_json = {
        "instruction": "max_overlap_alg",
        "input": {
            "timestamp": "06 Jun 2025 04:01:20.000",
            "sat_attrs": [
                {"id": 134, "health": 10.0, "pos": [7295.183, -566.176, -2919.725]},
                {"id": 143, "health": 10.0, "pos": [6050.902, 3968.745, 3114.593]},
                {"id": 166, "health": 10.0, "pos": [7475.208, -2377.897, -729.32]}
            ],
            "sat_edges": [
                {"from_sat": 134, "to_sat": 111, "distance": 7401.46},
                {"from_sat": 134, "to_sat": 116, "distance": 4104.49},
                {"from_sat": 134, "to_sat": 124, "distance": 8112.04},
                {"from_sat": 134, "to_sat": 125, "distance": 5420.33},
                {"from_sat": 134, "to_sat": 133, "distance": 7878.14},
                {"from_sat": 134, "to_sat": 135, "distance": 7878.14},
                {"from_sat": 134, "to_sat": 143, "distance": 7650.28},
                {"from_sat": 134, "to_sat": 144, "distance": 7694.04},
                {"from_sat": 134, "to_sat": 161, "distance": 9176.57},
                {"from_sat": 134, "to_sat": 165, "distance": 6930.6},
                {"from_sat": 134, "to_sat": 166, "distance": 2848.27},
                {"from_sat": 143, "to_sat": 111, "distance": 3195.72},
                {"from_sat": 143, "to_sat": 112, "distance": 6639.76},
                {"from_sat": 143, "to_sat": 133, "distance": 7607.16},
                {"from_sat": 143, "to_sat": 134, "distance": 7650.28},
                {"from_sat": 143, "to_sat": 142, "distance": 7878.14},
                {"from_sat": 143, "to_sat": 144, "distance": 7878.14},
                {"from_sat": 143, "to_sat": 152, "distance": 5279.04},
                {"from_sat": 143, "to_sat": 153, "distance": 8051.0},
                {"from_sat": 143, "to_sat": 161, "distance": 3813.7},
                {"from_sat": 143, "to_sat": 162, "distance": 9032.73},
                {"from_sat": 143, "to_sat": 166, "distance": 7555.41},
                {"from_sat": 166, "to_sat": 111, "distance": 8442.32},
                {"from_sat": 166, "to_sat": 116, "distance": 6717.36},
                {"from_sat": 166, "to_sat": 124, "distance": 6815.71},
                {"from_sat": 166, "to_sat": 125, "distance": 7320.13},
                {"from_sat": 166, "to_sat": 133, "distance": 5369.65},
                {"from_sat": 166, "to_sat": 134, "distance": 2848.27},
                {"from_sat": 166, "to_sat": 142, "distance": 9175.66},
                {"from_sat": 166, "to_sat": 143, "distance": 7555.41},
                {"from_sat": 166, "to_sat": 151, "distance": 7284.18},
                {"from_sat": 166, "to_sat": 156, "distance": 8335.09},
                {"from_sat": 166, "to_sat": 161, "distance": 7878.14},
                {"from_sat": 166, "to_sat": 165, "distance": 7878.14}
            ],
            "target_edges": [
                {"sat_id": 134, "target_id": 1, "quality": 0.1},
                {"sat_id": 143, "target_id": 1, "quality": 0.1},
                {"sat_id": 166, "target_id": 1, "quality": 0.1}
            ],
            "history_cluster_result": [[{
                "timestamp": "06 Jun 2025 04:01:10.000",
                "cluster_id": 0,
                "master": 143,
                "sats": [143],
                "targets": [1]
            }]]
        },
        "response": {
            "chain_of_thought": None,
            "clusters": [{
                "timestamp": "06 Jun 2025 04:01:20.000",
                "cluster_id": 0,
                "master": 134,
                "sats": [134, 143, 166],
                "targets": [1]
            }]
        }
    }
    
    # Create test conversation
    test_conversation = LLMConversationMessage(**test_data_json)
    
    # Initialize validator
    validator = ClusterDataValidator()
    
    print("Running cluster validation test...")
    print("=" * 50)
    
    # Run validation
    validation_results = validator.validate_output([test_conversation])
    
    # Display results
    print(f"Validation completed for {len(validation_results)} samples.")
    
    for i, result in enumerate(validation_results):
        print(f"Sample {i+1}:")
        print(f"  Total Score: {result.score:.2f}/100")
        print(f"  Timestamp: {result.input.timestamp}")
        print("  Detailed Results:")
        
        for detail in result.validation_details:
            print(f"    {detail.validation_type}: {detail.score}/100")
            if detail.info:
                info_lines = detail.info.split('\n')
                for line in info_lines:
                    if line.strip():
                        print(f"      {line}")
        print()
    
    print("Test completed!")
