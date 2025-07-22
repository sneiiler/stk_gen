"""
测试分簇稳定性验证函数

该测试文件针对 `_validate_stability_for_single_slice` 函数进行全面测试，
验证函数在各种稳定性场景下的表现。

使用方法:
    直接运行此文件即可执行所有测试:
    ```bash
    python test_stability_validation.py
    ```

测试场景包括:
    1. 无历史数据 - 验证满分情况
    2. 完全稳定 - 验证分簇完全一致的情况
    3. 轻微变化 - 验证少量目标/卫星切换
    4. 中等变化 - 验证适度的分簇调整
    5. 重大变化 - 验证大幅度分簇重组
    6. 完全重组 - 验证完全不同的分簇策略
    7. 簇数量变化 - 验证簇数量增减的情况
    8. 合理目标切换 - 验证因可见性变化导致的目标切换豁免
    9. 合理卫星切换 - 验证因连通性变化导致的卫星切换豁免
"""

import sys
from pathlib import Path
from typing import List, Optional

# 添加项目根目录到路径
root_dir = Path(__file__).parent
sys.path.append(str(root_dir))

from data_classes.sft_data_models import (
    LLMConversationMessage,
    RawConstellationDataModel,
    SatelliteClusterOutput,
    SatelliteAttributes,
    SatelliteEdge,
    TargetEdge,
    ClusterInfo
)
from distill._3_ClusterDataValidator import ClusterDataValidator


def create_test_conversation(
    sat_attrs: List[SatelliteAttributes],
    sat_edges: List[SatelliteEdge],
    target_edges: List[TargetEdge],
    clusters: List[ClusterInfo],
    history_cluster_result: Optional[List[List[ClusterInfo]]] = None
) -> LLMConversationMessage:
    """创建测试用的对话数据"""
    
    input_data = RawConstellationDataModel(
        timestamp="2025-07-21T10:00:00Z",
        sat_attrs=sat_attrs,
        sat_edges=sat_edges,
        target_edges=target_edges,
        history_cluster_result=history_cluster_result or []
    )
    
    response = SatelliteClusterOutput(
        chain_of_thought="测试推理过程",
        clusters=clusters
    )
    
    return LLMConversationMessage(
        instruction="进行卫星分簇",
        input=input_data,
        response=response
    )


def test_no_history_data():
    """测试无历史数据场景 - 应该得到满分100分"""
    print("🧪 测试场景1: 无历史数据 (期望得分: 100分)")
    
    # 创建基础卫星和目标数据
    sat_attrs = [
        SatelliteAttributes(id=1, health=10.0, pos=[1000, 2000, 3000]),
        SatelliteAttributes(id=2, health=9.5, pos=[1100, 2100, 3100]),
        SatelliteAttributes(id=3, health=9.0, pos=[1200, 2200, 3200]),
        SatelliteAttributes(id=4, health=8.5, pos=[2000, 3000, 4000])
    ]
    
    sat_edges = [
        SatelliteEdge(from_sat=1, to_sat=2, distance=100.0),
        SatelliteEdge(from_sat=2, to_sat=3, distance=110.0),
        SatelliteEdge(from_sat=1, to_sat=4, distance=200.0)
    ]
    
    target_edges = [
        TargetEdge(sat_id=1, target_id=101, quality=0.9),
        TargetEdge(sat_id=2, target_id=102, quality=0.8),
        TargetEdge(sat_id=3, target_id=103, quality=0.85),
        TargetEdge(sat_id=4, target_id=104, quality=0.75)
    ]
    
    clusters = [
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=0,
            master=1,
            sats=[1, 2],
            targets=[101, 102]
        ),
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=1,
            master=3,
            sats=[3, 4],
            targets=[103, 104]
        )
    ]
    
    # 无历史数据
    conversation = create_test_conversation(sat_attrs, sat_edges, target_edges, clusters, [])
    
    validator = ClusterDataValidator()
    result = validator._validate_stability_for_single_slice(conversation)
    
    print(f"   结果: {result.score}分")
    print(f"   详情: {result.info}")
    print()
    
    return result


def test_perfectly_stable():
    """测试完全稳定场景 - 分簇完全一致"""
    print("🧪 测试场景2: 完全稳定 (期望得分: 100分)")
    
    sat_attrs = [
        SatelliteAttributes(id=1, health=10.0, pos=[1000, 2000, 3000]),
        SatelliteAttributes(id=2, health=9.5, pos=[1100, 2100, 3100]),
        SatelliteAttributes(id=3, health=9.0, pos=[1200, 2200, 3200]),
        SatelliteAttributes(id=4, health=8.5, pos=[2000, 3000, 4000])
    ]
    
    sat_edges = [
        SatelliteEdge(from_sat=1, to_sat=2, distance=100.0),
        SatelliteEdge(from_sat=2, to_sat=3, distance=110.0),
        SatelliteEdge(from_sat=1, to_sat=4, distance=200.0)
    ]
    
    target_edges = [
        TargetEdge(sat_id=1, target_id=101, quality=0.9),
        TargetEdge(sat_id=2, target_id=102, quality=0.8),
        TargetEdge(sat_id=3, target_id=103, quality=0.85),
        TargetEdge(sat_id=4, target_id=104, quality=0.75)
    ]
    
    # 历史分簇（上一次的结果）
    history_clusters = [
        ClusterInfo(
            timestamp="2025-07-21T09:00:00Z",
            cluster_id=0,
            master=1,
            sats=[1, 2],
            targets=[101, 102]
        ),
        ClusterInfo(
            timestamp="2025-07-21T09:00:00Z",
            cluster_id=1,
            master=3,
            sats=[3, 4],
            targets=[103, 104]
        )
    ]
    
    # 当前分簇（完全相同）
    current_clusters = [
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=0,
            master=1,
            sats=[1, 2],
            targets=[101, 102]
        ),
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=1,
            master=3,
            sats=[3, 4],
            targets=[103, 104]
        )
    ]
    
    conversation = create_test_conversation(
        sat_attrs, sat_edges, target_edges, current_clusters, [history_clusters]
    )
    
    validator = ClusterDataValidator()
    result = validator._validate_stability_for_single_slice(conversation)
    
    print(f"   结果: {result.score}分")
    print(f"   详情: {result.info}")
    print()
    
    return result


def test_minor_changes():
    """测试轻微变化场景 - 少量目标/卫星切换"""
    print("🧪 测试场景3: 轻微变化 (期望得分: 高分)")
    
    sat_attrs = [
        SatelliteAttributes(id=1, health=10.0, pos=[1000, 2000, 3000]),
        SatelliteAttributes(id=2, health=9.5, pos=[1100, 2100, 3100]),
        SatelliteAttributes(id=3, health=9.0, pos=[1200, 2200, 3200]),
        SatelliteAttributes(id=4, health=8.5, pos=[2000, 3000, 4000]),
        SatelliteAttributes(id=5, health=8.0, pos=[2100, 3100, 4100])
    ]
    
    sat_edges = [
        SatelliteEdge(from_sat=1, to_sat=2, distance=100.0),
        SatelliteEdge(from_sat=2, to_sat=3, distance=110.0),
        SatelliteEdge(from_sat=3, to_sat=4, distance=120.0),
        SatelliteEdge(from_sat=4, to_sat=5, distance=130.0)
    ]
    
    target_edges = [
        TargetEdge(sat_id=1, target_id=101, quality=0.9),
        TargetEdge(sat_id=2, target_id=102, quality=0.8),
        TargetEdge(sat_id=3, target_id=103, quality=0.85),
        TargetEdge(sat_id=4, target_id=104, quality=0.75),
        TargetEdge(sat_id=5, target_id=105, quality=0.8)
    ]
    
    # 历史分簇
    history_clusters = [
        ClusterInfo(
            timestamp="2025-07-21T09:00:00Z",
            cluster_id=0,
            master=1,
            sats=[1, 2, 3],
            targets=[101, 102, 103]
        ),
        ClusterInfo(
            timestamp="2025-07-21T09:00:00Z",
            cluster_id=1,
            master=4,
            sats=[4, 5],
            targets=[104, 105]
        )
    ]
    
    # 当前分簇（轻微变化：卫星3从簇0移到簇1，但大部分保持不变）
    current_clusters = [
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=0,
            master=1,
            sats=[1, 2],  # 卫星3移除
            targets=[101, 102]  # 目标103移除
        ),
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=1,
            master=4,
            sats=[3, 4, 5],  # 卫星3加入
            targets=[103, 104, 105]  # 目标103加入
        )
    ]
    
    conversation = create_test_conversation(
        sat_attrs, sat_edges, target_edges, current_clusters, [history_clusters]
    )
    
    validator = ClusterDataValidator()
    result = validator._validate_stability_for_single_slice(conversation)
    
    print(f"   结果: {result.score}分")
    print(f"   详情: {result.info}")
    print()
    
    return result


def test_moderate_changes():
    """测试中等变化场景 - 适度的分簇调整"""
    print("🧪 测试场景4: 中等变化 (期望得分: 中等)")
    
    sat_attrs = [
        SatelliteAttributes(id=1, health=10.0, pos=[1000, 2000, 3000]),
        SatelliteAttributes(id=2, health=9.5, pos=[1100, 2100, 3100]),
        SatelliteAttributes(id=3, health=9.0, pos=[1200, 2200, 3200]),
        SatelliteAttributes(id=4, health=8.5, pos=[2000, 3000, 4000]),
        SatelliteAttributes(id=5, health=8.0, pos=[2100, 3100, 4100]),
        SatelliteAttributes(id=6, health=7.5, pos=[2200, 3200, 4200])
    ]
    
    sat_edges = [
        SatelliteEdge(from_sat=1, to_sat=2, distance=100.0),
        SatelliteEdge(from_sat=2, to_sat=3, distance=110.0),
        SatelliteEdge(from_sat=3, to_sat=4, distance=120.0),
        SatelliteEdge(from_sat=4, to_sat=5, distance=130.0),
        SatelliteEdge(from_sat=5, to_sat=6, distance=140.0)
    ]
    
    target_edges = [
        TargetEdge(sat_id=1, target_id=101, quality=0.9),
        TargetEdge(sat_id=2, target_id=102, quality=0.8),
        TargetEdge(sat_id=3, target_id=103, quality=0.85),
        TargetEdge(sat_id=4, target_id=104, quality=0.75),
        TargetEdge(sat_id=5, target_id=105, quality=0.8),
        TargetEdge(sat_id=6, target_id=106, quality=0.9)
    ]
    
    # 历史分簇（三个簇）
    history_clusters = [
        ClusterInfo(
            timestamp="2025-07-21T09:00:00Z",
            cluster_id=0,
            master=1,
            sats=[1, 2],
            targets=[101, 102]
        ),
        ClusterInfo(
            timestamp="2025-07-21T09:00:00Z",
            cluster_id=1,
            master=3,
            sats=[3, 4],
            targets=[103, 104]
        ),
        ClusterInfo(
            timestamp="2025-07-21T09:00:00Z",
            cluster_id=2,
            master=5,
            sats=[5, 6],
            targets=[105, 106]
        )
    ]
    
    # 当前分簇（重新组织：将三个簇合并为两个）
    current_clusters = [
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=0,
            master=1,
            sats=[1, 2, 3],  # 增加卫星3
            targets=[101, 102, 103]  # 增加目标103
        ),
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=1,
            master=4,
            sats=[4, 5, 6],  # 卫星4,5,6重新组合
            targets=[104, 105, 106]  # 目标104,105,106重新组合
        )
    ]
    
    conversation = create_test_conversation(
        sat_attrs, sat_edges, target_edges, current_clusters, [history_clusters]
    )
    
    validator = ClusterDataValidator()
    result = validator._validate_stability_for_single_slice(conversation)
    
    print(f"   结果: {result.score}分")
    print(f"   详情: {result.info}")
    print()
    
    return result


def test_major_changes():
    """测试重大变化场景 - 大幅度分簇重组"""
    print("🧪 测试场景5: 重大变化 (期望扣分)")
    
    sat_attrs = [
        SatelliteAttributes(id=1, health=10.0, pos=[1000, 2000, 3000]),
        SatelliteAttributes(id=2, health=9.5, pos=[1100, 2100, 3100]),
        SatelliteAttributes(id=3, health=9.0, pos=[1200, 2200, 3200]),
        SatelliteAttributes(id=4, health=8.5, pos=[2000, 3000, 4000]),
        SatelliteAttributes(id=5, health=8.0, pos=[2100, 3100, 4100]),
        SatelliteAttributes(id=6, health=7.5, pos=[2200, 3200, 4200])
    ]
    
    sat_edges = [
        SatelliteEdge(from_sat=1, to_sat=2, distance=100.0),
        SatelliteEdge(from_sat=2, to_sat=3, distance=110.0),
        SatelliteEdge(from_sat=3, to_sat=4, distance=120.0),
        SatelliteEdge(from_sat=4, to_sat=5, distance=130.0),
        SatelliteEdge(from_sat=5, to_sat=6, distance=140.0)
    ]
    
    target_edges = [
        TargetEdge(sat_id=1, target_id=101, quality=0.9),
        TargetEdge(sat_id=2, target_id=102, quality=0.8),
        TargetEdge(sat_id=3, target_id=103, quality=0.85),
        TargetEdge(sat_id=4, target_id=104, quality=0.75),
        TargetEdge(sat_id=5, target_id=105, quality=0.8),
        TargetEdge(sat_id=6, target_id=106, quality=0.9)
    ]
    
    # 历史分簇
    history_clusters = [
        ClusterInfo(
            timestamp="2025-07-21T09:00:00Z",
            cluster_id=0,
            master=1,
            sats=[1, 2, 3],
            targets=[101, 102, 103]
        ),
        ClusterInfo(
            timestamp="2025-07-21T09:00:00Z",
            cluster_id=1,
            master=4,
            sats=[4, 5, 6],
            targets=[104, 105, 106]
        )
    ]
    
    # 当前分簇（大幅度变化：大多数元素都切换了簇）
    current_clusters = [
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=0,
            master=1,
            sats=[1, 4, 5],  # 卫星4,5从簇1移到簇0
            targets=[101, 104, 105]  # 目标104,105从簇1移到簇0
        ),
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=1,
            master=2,
            sats=[2, 3, 6],  # 卫星2,3从簇0移到簇1，卫星6留在簇1
            targets=[102, 103, 106]  # 目标102,103从簇0移到簇1，目标106留在簇1
        )
    ]
    
    conversation = create_test_conversation(
        sat_attrs, sat_edges, target_edges, current_clusters, [history_clusters]
    )
    
    validator = ClusterDataValidator()
    result = validator._validate_stability_for_single_slice(conversation)
    
    print(f"   结果: {result.score}分")
    print(f"   详情: {result.info}")
    print()
    
    return result


def test_complete_reorganization():
    """测试完全重组场景 - 完全不同的分簇策略"""
    print("🧪 测试场景6: 完全重组 (期望大幅扣分)")
    
    sat_attrs = [
        SatelliteAttributes(id=1, health=10.0, pos=[1000, 2000, 3000]),
        SatelliteAttributes(id=2, health=9.5, pos=[1100, 2100, 3100]),
        SatelliteAttributes(id=3, health=9.0, pos=[1200, 2200, 3200]),
        SatelliteAttributes(id=4, health=8.5, pos=[2000, 3000, 4000]),
        SatelliteAttributes(id=5, health=8.0, pos=[2100, 3100, 4100]),
        SatelliteAttributes(id=6, health=7.5, pos=[2200, 3200, 4200])
    ]
    
    sat_edges = [
        SatelliteEdge(from_sat=1, to_sat=2, distance=100.0),
        SatelliteEdge(from_sat=2, to_sat=3, distance=110.0),
        SatelliteEdge(from_sat=3, to_sat=4, distance=120.0),
        SatelliteEdge(from_sat=4, to_sat=5, distance=130.0),
        SatelliteEdge(from_sat=5, to_sat=6, distance=140.0)
    ]
    
    target_edges = [
        TargetEdge(sat_id=1, target_id=101, quality=0.9),
        TargetEdge(sat_id=2, target_id=102, quality=0.8),
        TargetEdge(sat_id=3, target_id=103, quality=0.85),
        TargetEdge(sat_id=4, target_id=104, quality=0.75),
        TargetEdge(sat_id=5, target_id=105, quality=0.8),
        TargetEdge(sat_id=6, target_id=106, quality=0.9)
    ]
    
    # 历史分簇（两个大簇）
    history_clusters = [
        ClusterInfo(
            timestamp="2025-07-21T09:00:00Z",
            cluster_id=0,
            master=1,
            sats=[1, 2, 3],
            targets=[101, 102, 103]
        ),
        ClusterInfo(
            timestamp="2025-07-21T09:00:00Z",
            cluster_id=1,
            master=4,
            sats=[4, 5, 6],
            targets=[104, 105, 106]
        )
    ]
    
    # 当前分簇（完全不同：每个卫星单独成簇）
    current_clusters = [
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=0,
            master=1,
            sats=[1],
            targets=[101]
        ),
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=1,
            master=2,
            sats=[2],
            targets=[102]
        ),
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=2,
            master=3,
            sats=[3],
            targets=[103]
        ),
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=3,
            master=4,
            sats=[4],
            targets=[104]
        ),
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=4,
            master=5,
            sats=[5],
            targets=[105]
        ),
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=5,
            master=6,
            sats=[6],
            targets=[106]
        )
    ]
    
    conversation = create_test_conversation(
        sat_attrs, sat_edges, target_edges, current_clusters, [history_clusters]
    )
    
    validator = ClusterDataValidator()
    result = validator._validate_stability_for_single_slice(conversation)
    
    print(f"   结果: {result.score}分")
    print(f"   详情: {result.info}")
    print()
    
    return result


def test_cluster_count_change():
    """测试簇数量变化场景 - 簇数量增减的情况"""
    print("🧪 测试场景7: 簇数量变化 (期望扣分)")
    
    sat_attrs = [
        SatelliteAttributes(id=1, health=10.0, pos=[1000, 2000, 3000]),
        SatelliteAttributes(id=2, health=9.5, pos=[1100, 2100, 3100]),
        SatelliteAttributes(id=3, health=9.0, pos=[1200, 2200, 3200]),
        SatelliteAttributes(id=4, health=8.5, pos=[2000, 3000, 4000]),
        SatelliteAttributes(id=5, health=8.0, pos=[2100, 3100, 4100]),
        SatelliteAttributes(id=6, health=7.5, pos=[2200, 3200, 4200])
    ]
    
    sat_edges = [
        SatelliteEdge(from_sat=1, to_sat=2, distance=100.0),
        SatelliteEdge(from_sat=2, to_sat=3, distance=110.0),
        SatelliteEdge(from_sat=3, to_sat=4, distance=120.0),
        SatelliteEdge(from_sat=4, to_sat=5, distance=130.0),
        SatelliteEdge(from_sat=5, to_sat=6, distance=140.0)
    ]
    
    target_edges = [
        TargetEdge(sat_id=1, target_id=101, quality=0.9),
        TargetEdge(sat_id=2, target_id=102, quality=0.8),
        TargetEdge(sat_id=3, target_id=103, quality=0.85),
        TargetEdge(sat_id=4, target_id=104, quality=0.75),
        TargetEdge(sat_id=5, target_id=105, quality=0.8),
        TargetEdge(sat_id=6, target_id=106, quality=0.9)
    ]
    
    # 历史分簇（三个簇）
    history_clusters = [
        ClusterInfo(
            timestamp="2025-07-21T09:00:00Z",
            cluster_id=0,
            master=1,
            sats=[1, 2],
            targets=[101, 102]
        ),
        ClusterInfo(
            timestamp="2025-07-21T09:00:00Z",
            cluster_id=1,
            master=3,
            sats=[3, 4],
            targets=[103, 104]
        ),
        ClusterInfo(
            timestamp="2025-07-21T09:00:00Z",
            cluster_id=2,
            master=5,
            sats=[5, 6],
            targets=[105, 106]
        )
    ]
    
    # 当前分簇（减少到一个大簇）
    current_clusters = [
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=0,
            master=1,
            sats=[1, 2, 3, 4, 5, 6],  # 所有卫星合并
            targets=[101, 102, 103, 104, 105, 106]  # 所有目标合并
        )
    ]
    
    conversation = create_test_conversation(
        sat_attrs, sat_edges, target_edges, current_clusters, [history_clusters]
    )
    
    validator = ClusterDataValidator()
    result = validator._validate_stability_for_single_slice(conversation)
    
    print(f"   结果: {result.score}分")
    print(f"   详情: {result.info}")
    print()
    
    return result


def test_justified_target_switch():
    """测试合理的目标切换场景（因可见性变化） - 应该豁免扣分"""
    print("🧪 测试场景8: 合理的目标切换 (期望豁免，得分: 100分)")
    
    sat_attrs = [
        SatelliteAttributes(id=1, health=10.0, pos=[1000, 2000, 3000]),
        SatelliteAttributes(id=2, health=9.5, pos=[1100, 2100, 3100]),
        SatelliteAttributes(id=3, health=9.0, pos=[1200, 2200, 3200]),
    ]
    
    sat_edges = [
        SatelliteEdge(from_sat=1, to_sat=2, distance=100.0),
        SatelliteEdge(from_sat=2, to_sat=3, distance=110.0),
    ]
    
    # 历史分簇
    history_clusters = [
        ClusterInfo(
            timestamp="2025-07-21T09:00:00Z",
            cluster_id=0,
            master=1,
            sats=[1, 2],
            targets=[101, 102]
        ),
        ClusterInfo(
            timestamp="2025-07-21T09:00:00Z",
            cluster_id=1,
            master=3,
            sats=[3],
            targets=[103]
        )
    ]
    
    # 当前可见性：目标102已经无法被原簇0的卫星(1,2)观测，只能被簇1的卫星3观测
    target_edges = [
        TargetEdge(sat_id=1, target_id=101, quality=0.9),
        TargetEdge(sat_id=3, target_id=102, quality=0.8), # 目标102的可见性变化
        TargetEdge(sat_id=3, target_id=103, quality=0.85),
    ]
    
    # 当前分簇：目标102从簇0切换到簇1
    current_clusters = [
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=0,
            master=1,
            sats=[1, 2],
            targets=[101] # 目标102移除
        ),
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=1,
            master=3,
            sats=[3],
            targets=[102, 103] # 目标102加入
        )
    ]
    
    conversation = create_test_conversation(
        sat_attrs, sat_edges, target_edges, current_clusters, [history_clusters]
    )
    
    validator = ClusterDataValidator()
    result = validator._validate_stability_for_single_slice(conversation)
    
    print(f"   结果: {result.score}分")
    print(f"   详情: {result.info}")
    print()
    
    return result


def test_justified_satellite_switch():
    """测试合理的卫星切换场景（因连通性变化） - 应该豁免扣分"""
    print("🧪 测试场景9: 合理的卫星切换 (期望豁免，得分: 100分)")
    
    sat_attrs = [
        SatelliteAttributes(id=1, health=10.0, pos=[1000, 2000, 3000]),
        SatelliteAttributes(id=2, health=9.5, pos=[1100, 2100, 3100]),
        SatelliteAttributes(id=3, health=9.0, pos=[1200, 2200, 3200]),
    ]
    
    # 历史分簇
    history_clusters = [
        ClusterInfo(
            timestamp="2025-07-21T09:00:00Z",
            cluster_id=0,
            master=1,
            sats=[1, 2],
            targets=[101, 102]
        ),
        ClusterInfo(
            timestamp="2025-07-21T09:00:00Z",
            cluster_id=1,
            master=3,
            sats=[3],
            targets=[103]
        )
    ]
    
    # 当前连通性：卫星2与原簇0的卫星1失去连接，但与簇1的卫星3建立了连接
    sat_edges = [
        SatelliteEdge(from_sat=2, to_sat=3, distance=110.0),
    ]
    
    target_edges = [
        TargetEdge(sat_id=1, target_id=101, quality=0.9),
        TargetEdge(sat_id=2, target_id=102, quality=0.8),
        TargetEdge(sat_id=3, target_id=103, quality=0.85),
    ]
    
    # 当前分簇：卫星2从簇0切换到簇1
    current_clusters = [
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=0,
            master=1,
            sats=[1], # 卫星2移除
            targets=[101]
        ),
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=1,
            master=3,
            sats=[2, 3], # 卫星2加入
            targets=[102, 103]
        )
    ]
    
    conversation = create_test_conversation(
        sat_attrs, sat_edges, target_edges, current_clusters, [history_clusters]
    )
    
    validator = ClusterDataValidator()
    result = validator._validate_stability_for_single_slice(conversation)
    
    print(f"   结果: {result.score}分")
    print(f"   详情: {result.info}")
    print()
    
    return result


def run_all_tests():
    """运行所有测试"""
    print("=" * 80)
    print("🚀 开始运行分簇稳定性验证函数测试")
    print("=" * 80)
    
    results = []
    
    # 运行所有测试场景
    results.append(test_no_history_data())
    results.append(test_perfectly_stable())
    results.append(test_minor_changes())
    results.append(test_moderate_changes())
    results.append(test_major_changes())
    results.append(test_complete_reorganization())
    results.append(test_cluster_count_change())
    results.append(test_justified_target_switch())
    results.append(test_justified_satellite_switch())
    
    # 统计测试结果
    print("=" * 80)
    print("📊 测试结果汇总")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        validation_type = result.validation_type
        score = result.score
        
        # 提取关键信息
        if "ERROR" in result.info:
            status = "❌ 发现错误"
        elif "WARNING" in result.info:
            status = "⚠️  有警告"
        else:
            status = "✅ 正常"
        
        print(f"测试{i}: {score}/100分 - {status}")
    
    print("\n🎯 测试完成！")
    print("这些测试验证了分簇稳定性验证函数在各种场景下的表现:")
    print("• 无历史数据：验证缺少历史数据时的处理")
    print("• 完全稳定：验证完全一致分簇的满分情况")
    print("• 轻微变化：验证少量调整的容忍度")
    print("• 中等变化：验证适度重组的扣分机制")
    print("• 重大变化：验证大幅变化的重度扣分")
    print("• 完全重组：验证完全不同策略的严厉扣分")
    print("• 簇数量变化：验证簇数量变化的影响")
    print("• 合理目标切换：验证因可见性变化导致的目标切换豁免")
    print("• 合理卫星切换：验证因连通性变化导致的卫星切换豁免")


if __name__ == "__main__":
    run_all_tests()
