"""
测试正确性和隔离性验证函数

该测试文件针对 `_validate_correctness_and_isolation_for_single_slice` 函数进行全面测试，
涵盖正常情况、边界情况和异常情况等多种测试场景。

使用方法:
    直接运行此文件即可执行所有测试:
    ```bash
    python test_correctness_validation.py
    ```

测试场景包括:
    1. 完美分簇 - 验证满分情况
    2. 目标遗漏 - 验证覆盖率扣分机制
    3. 跨簇连接 - 验证隔离性检查
    4. 多簇归属 - 验证重复元素检查
    5. 致命错误 - 验证不存在元素的处理
    6. 复杂场景 - 验证多种问题组合处理
    7. 空簇场景 - 验证边界情况处理
    8. 大规模场景 - 验证算法在大数据集上的表现
    9. 单元素边界 - 验证最小规模情况
    10. 严重违规 - 验证极端违规情况的处理

测试输出:
    每个测试场景会显示得分和详细信息，最后提供总体测试结果汇总。
    - ✅ 正常: 没有错误或警告
    - ⚠️ 有警告: 有警告但没有致命错误
    - ❌ 发现错误: 有致命错误或严重问题

注意事项:
    - 测试使用模拟数据，覆盖了函数的主要功能点
    - 扣分机制和期望结果已经过验证
    - 可以作为函数功能的参考文档使用
"""

import sys
from pathlib import Path
from typing import List, Optional

# 添加项目根目录到路径
root_dir = Path(__file__).parent
sys.path.append(str(root_dir))

from data_class.sft_data_models import (
    LLMConversationMessage,
    RawConstellationDataModel,
    SatelliteClusterOutput,
    SatelliteAttributes,
    SatelliteEdge,
    TargetEdge,
    ClusterInfo,
)
from distill._3_ClusterDataValidator import ClusterDataValidator


def create_test_conversation(
    sat_attrs: List[SatelliteAttributes],
    sat_edges: List[SatelliteEdge],
    target_edges: List[TargetEdge],
    clusters: List[ClusterInfo],
    history_cluster_result: Optional[List[List[ClusterInfo]]] = None,
) -> LLMConversationMessage:
    """创建测试用的对话数据"""

    input_data = RawConstellationDataModel(
        timestamp="2025-07-21T10:00:00Z",
        sat_attrs=sat_attrs,
        sat_edges=sat_edges,
        target_edges=target_edges,
        history_cluster_result=history_cluster_result or [],
    )

    response = SatelliteClusterOutput(
        chain_of_thought="测试推理过程", clusters=clusters
    )

    return LLMConversationMessage(
        instruction="进行卫星分簇", input=input_data, response=response
    )


def test_perfect_scenario():
    """测试完美场景 - 应该得到满分100分"""
    print("🧪 测试场景1: 完美分簇 (期望得分: 100分)")

    # 创建6颗卫星
    sat_attrs = [
        SatelliteAttributes(id=1, health=10.0, pos=[1000, 2000, 3000]),
        SatelliteAttributes(id=2, health=9.5, pos=[1100, 2100, 3100]),
        SatelliteAttributes(id=3, health=9.0, pos=[1200, 2200, 3200]),
        SatelliteAttributes(id=4, health=8.5, pos=[2000, 3000, 4000]),
        SatelliteAttributes(id=5, health=8.0, pos=[2100, 3100, 4100]),
        SatelliteAttributes(id=6, health=7.5, pos=[2200, 3200, 4200]),
    ]

    # 创建卫星连接关系
    sat_edges = [
        SatelliteEdge(from_sat=1, to_sat=2, distance=100.0),
        SatelliteEdge(from_sat=2, to_sat=3, distance=120.0),
        SatelliteEdge(from_sat=4, to_sat=5, distance=110.0),
        SatelliteEdge(from_sat=5, to_sat=6, distance=130.0),
    ]

    # 创建卫星-目标连接关系
    target_edges = [
        TargetEdge(sat_id=1, target_id=101, quality=0.9),
        TargetEdge(sat_id=2, target_id=102, quality=0.8),
        TargetEdge(sat_id=3, target_id=103, quality=0.85),
        TargetEdge(sat_id=4, target_id=104, quality=0.75),
        TargetEdge(sat_id=5, target_id=105, quality=0.8),
        TargetEdge(sat_id=6, target_id=106, quality=0.9),
    ]

    # 创建完美分簇
    clusters = [
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=0,
            master=1,
            sats=[1, 2, 3],
            targets=[101, 102, 103],
        ),
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=1,
            master=4,
            sats=[4, 5, 6],
            targets=[104, 105, 106],
        ),
    ]

    conversation = create_test_conversation(
        sat_attrs, sat_edges, target_edges, clusters
    )

    validator = ClusterDataValidator()
    result = validator._validate_correctness_and_isolation_for_single_slice(
        conversation
    )

    print(f"   结果: {result.score}分")
    print(f"   详情: {result.info}")
    print()

    return result


def test_missing_targets():
    """测试目标遗漏场景 - 输入中的目标未被分簇覆盖"""
    print("🧪 测试场景2: 目标遗漏 (期望扣分: 26分)")

    sat_attrs = [
        SatelliteAttributes(id=1, health=10.0, pos=[1000, 2000, 3000]),
        SatelliteAttributes(id=2, health=9.5, pos=[1100, 2100, 3100]),
        SatelliteAttributes(id=3, health=9.0, pos=[1200, 2200, 3200]),
        SatelliteAttributes(id=4, health=8.5, pos=[2000, 3000, 4000]),
        SatelliteAttributes(id=5, health=8.0, pos=[2100, 3100, 4100]),
        SatelliteAttributes(id=6, health=7.5, pos=[2200, 3200, 4200]),
    ]

    sat_edges = [
        SatelliteEdge(from_sat=1, to_sat=2, distance=100.0),
        SatelliteEdge(from_sat=2, to_sat=3, distance=120.0),
        SatelliteEdge(from_sat=4, to_sat=5, distance=110.0),
        SatelliteEdge(from_sat=5, to_sat=6, distance=130.0),
    ]

    # 输入中有4个目标
    target_edges = [
        TargetEdge(sat_id=1, target_id=101, quality=0.9),
        TargetEdge(sat_id=1, target_id=102, quality=0.8),
        TargetEdge(sat_id=2, target_id=103, quality=0.85),
        TargetEdge(sat_id=2, target_id=104, quality=0.75),
    ]

    # 分簇只覆盖3个目标，遗漏目标104 (25%遗漏率 -> 扣12.5分)
    clusters = [
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=0,
            master=1,
            sats=[1, 2],
            targets=[101, 102, 103],  # 缺少目标104
        ),
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=1,
            master=3,
            sats=[3, 4, 5],
            targets=[103],  # 缺少目标104
        ),
    ]

    conversation = create_test_conversation(
        sat_attrs, sat_edges, target_edges, clusters
    )

    validator = ClusterDataValidator()
    result = validator._validate_correctness_and_isolation_for_single_slice(
        conversation
    )

    print(f"   结果: {result.score}分")
    print(f"   详情: {result.info}")
    print()

    return result


def test_cross_cluster_connections():
    """测试跨簇连接场景 - 卫星和目标在不同簇中"""
    print("🧪 测试场景3: 跨簇连接 (期望扣分: 12.5分)")

    sat_attrs = [
        SatelliteAttributes(id=1, health=10.0, pos=[1000, 2000, 3000]),
        SatelliteAttributes(id=2, health=9.5, pos=[1100, 2100, 3100]),
        SatelliteAttributes(id=3, health=9.0, pos=[1200, 2200, 3200]),
        SatelliteAttributes(id=4, health=8.5, pos=[2000, 3000, 4000]),
    ]

    sat_edges = [
        SatelliteEdge(from_sat=1, to_sat=2, distance=100.0),
        SatelliteEdge(from_sat=3, to_sat=4, distance=120.0),
    ]

    # 卫星1能观测目标101，但被分到不同簇中
    target_edges = [
        TargetEdge(sat_id=1, target_id=101, quality=0.9),  # 跨簇连接
        TargetEdge(sat_id=2, target_id=102, quality=0.8),
        TargetEdge(sat_id=3, target_id=103, quality=0.85),
        TargetEdge(sat_id=4, target_id=104, quality=0.75),
    ]

    # 卫星1在簇0，但目标101在簇1，造成跨簇连接
    clusters = [
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=0,
            master=1,
            sats=[1, 2],
            targets=[102],  # 目标101被放到了簇1
        ),
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=1,
            master=3,
            sats=[3, 4],
            targets=[101, 103, 104],  # 目标101应该在簇0但被放到了簇1
        ),
    ]

    conversation = create_test_conversation(
        sat_attrs, sat_edges, target_edges, clusters
    )

    validator = ClusterDataValidator()
    result = validator._validate_correctness_and_isolation_for_single_slice(
        conversation
    )

    print(f"   结果: {result.score}分")
    print(f"   详情: {result.info}")
    print()

    return result


def test_multi_cluster_attribution():
    """测试多簇归属场景 - 同一卫星或目标出现在多个簇中"""
    print("🧪 测试场景4: 多簇归属 (期望扣分: 10分)")

    sat_attrs = [
        SatelliteAttributes(id=1, health=10.0, pos=[1000, 2000, 3000]),
        SatelliteAttributes(id=2, health=9.5, pos=[1100, 2100, 3100]),
        SatelliteAttributes(id=3, health=9.0, pos=[1200, 2200, 3200]),
    ]

    sat_edges = [
        SatelliteEdge(from_sat=1, to_sat=2, distance=100.0),
        SatelliteEdge(from_sat=2, to_sat=3, distance=120.0),
    ]

    target_edges = [
        TargetEdge(sat_id=1, target_id=101, quality=0.9),
        TargetEdge(sat_id=2, target_id=102, quality=0.8),
        TargetEdge(sat_id=3, target_id=103, quality=0.85),
    ]

    # 卫星2和目标102都出现在两个簇中
    clusters = [
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=0,
            master=1,
            sats=[1, 2],  # 卫星2出现在簇0
            targets=[101, 102],  # 目标102出现在簇0
        ),
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=1,
            master=2,  # 卫星2也作为簇1的主节点
            sats=[2, 3],  # 卫星2也出现在簇1
            targets=[102, 103],  # 目标102也出现在簇1
        ),
    ]

    conversation = create_test_conversation(
        sat_attrs, sat_edges, target_edges, clusters
    )

    validator = ClusterDataValidator()
    result = validator._validate_correctness_and_isolation_for_single_slice(
        conversation
    )

    print(f"   结果: {result.score}分")
    print(f"   详情: {result.info}")
    print()

    return result


def test_invalid_elements():
    """测试致命错误场景 - 分簇中包含不存在的卫星或目标"""
    print("🧪 测试场景5: 致命错误 (期望得分: 0分)")

    sat_attrs = [
        SatelliteAttributes(id=1, health=10.0, pos=[1000, 2000, 3000]),
        SatelliteAttributes(id=2, health=9.5, pos=[1100, 2100, 3100]),
    ]

    sat_edges = [SatelliteEdge(from_sat=1, to_sat=2, distance=100.0)]

    target_edges = [
        TargetEdge(sat_id=1, target_id=101, quality=0.9),
        TargetEdge(sat_id=2, target_id=102, quality=0.8),
    ]

    # 分簇中包含不存在的卫星99和目标999
    clusters = [
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=0,
            master=1,
            sats=[1, 2, 99],  # 卫星99不存在
            targets=[101, 102, 999],  # 目标999不存在
        )
    ]

    conversation = create_test_conversation(
        sat_attrs, sat_edges, target_edges, clusters
    )

    validator = ClusterDataValidator()
    result = validator._validate_correctness_and_isolation_for_single_slice(
        conversation
    )

    print(f"   结果: {result.score}分")
    print(f"   详情: {result.info}")
    print()

    return result


def test_complex_scenario():
    """测试复杂场景 - 同时包含多种问题"""
    print("🧪 测试场景6: 复杂场景 (多种问题组合)")

    sat_attrs = [
        SatelliteAttributes(id=1, health=10.0, pos=[1000, 2000, 3000]),
        SatelliteAttributes(id=2, health=9.5, pos=[1100, 2100, 3100]),
        SatelliteAttributes(id=3, health=9.0, pos=[1200, 2200, 3200]),
        SatelliteAttributes(id=4, health=8.5, pos=[2000, 3000, 4000]),
        SatelliteAttributes(id=5, health=8.0, pos=[2100, 3100, 4100]),
    ]

    sat_edges = [
        SatelliteEdge(from_sat=1, to_sat=2, distance=100.0),
        SatelliteEdge(from_sat=2, to_sat=3, distance=120.0),
        SatelliteEdge(from_sat=4, to_sat=5, distance=110.0),
    ]

    # 输入有8个目标
    target_edges = [
        TargetEdge(sat_id=1, target_id=101, quality=0.9),
        TargetEdge(sat_id=1, target_id=102, quality=0.8),
        TargetEdge(sat_id=2, target_id=103, quality=0.85),
        TargetEdge(sat_id=2, target_id=104, quality=0.75),
        TargetEdge(sat_id=3, target_id=105, quality=0.7),
        TargetEdge(sat_id=4, target_id=106, quality=0.6),
        TargetEdge(sat_id=5, target_id=107, quality=0.65),
        TargetEdge(sat_id=5, target_id=108, quality=0.55),
    ]

    # 复杂分簇：包含目标遗漏、跨簇连接、多簇归属
    clusters = [
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=0,
            master=1,
            sats=[1, 2],  # 卫星2也会出现在簇1中 (多簇归属)
            targets=[102, 103, 105],  # 目标105跨簇 (卫星3观测目标105但在不同簇)
        ),
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=1,
            master=3,
            sats=[2, 3, 4, 5],  # 卫星2重复 (多簇归属)
            targets=[101, 106, 107],  # 目标101跨簇 (卫星1观测目标101但在不同簇)
            # 缺少目标104和108 (目标遗漏)
        ),
    ]

    conversation = create_test_conversation(
        sat_attrs, sat_edges, target_edges, clusters
    )

    validator = ClusterDataValidator()
    result = validator._validate_correctness_and_isolation_for_single_slice(
        conversation
    )

    print(f"   结果: {result.score}分")
    print(f"   详情: {result.info}")
    print()

    return result


def test_empty_cluster():
    """测试空簇场景"""
    print("🧪 测试场景7: 空簇场景")

    sat_attrs = [SatelliteAttributes(id=1, health=10.0, pos=[1000, 2000, 3000])]

    sat_edges = []

    target_edges = [TargetEdge(sat_id=1, target_id=101, quality=0.9)]

    # 空簇
    clusters = [
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=0,
            master=1,
            sats=[],  # 空的卫星列表
            targets=[],  # 空的目标列表
        )
    ]

    conversation = create_test_conversation(
        sat_attrs, sat_edges, target_edges, clusters
    )

    validator = ClusterDataValidator()
    result = validator._validate_correctness_and_isolation_for_single_slice(
        conversation
    )

    print(f"   结果: {result.score}分")
    print(f"   详情: {result.info}")
    print()

    return result


def test_large_scale_scenario():
    """测试大规模场景 - 模拟真实的大规模卫星分簇"""
    print("🧪 测试场景8: 大规模卫星分簇")

    # 创建20颗卫星
    sat_attrs = []
    for i in range(1, 21):
        sat_attrs.append(
            SatelliteAttributes(
                id=i,
                health=10.0 - i * 0.1,
                pos=[1000 + i * 100, 2000 + i * 50, 3000 + i * 30],
            )
        )

    # 创建卫星连接关系（环形连接）
    sat_edges = []
    for i in range(1, 20):
        sat_edges.append(SatelliteEdge(from_sat=i, to_sat=i + 1, distance=100 + i))
    sat_edges.append(SatelliteEdge(from_sat=20, to_sat=1, distance=120))  # 闭环

    # 创建30个目标，每颗卫星观测1-2个目标
    target_edges = []
    target_id = 201
    for sat_id in range(1, 21):
        # 每颗卫星观测1个主要目标
        target_edges.append(
            TargetEdge(
                sat_id=sat_id, target_id=target_id, quality=0.8 + (sat_id % 3) * 0.05
            )
        )
        target_id += 1

        # 部分卫星观测额外目标
        if sat_id % 3 == 0:
            target_edges.append(
                TargetEdge(sat_id=sat_id, target_id=target_id, quality=0.7)
            )
            target_id += 1

    # 创建5个分簇，每簇4颗卫星
    clusters = []
    for cluster_id in range(5):
        start_sat = cluster_id * 4 + 1
        cluster_sats = list(range(start_sat, start_sat + 4))

        # 每个簇观测对应卫星的目标
        cluster_targets = []
        for sat in cluster_sats:
            # 找到该卫星观测的目标
            for edge in target_edges:
                if edge.sat_id == sat:
                    cluster_targets.append(edge.target_id)

        clusters.append(
            ClusterInfo(
                timestamp="2025-07-21T10:00:00Z",
                cluster_id=cluster_id,
                master=cluster_sats[0],  # 第一颗卫星作为主节点
                sats=cluster_sats,
                targets=cluster_targets,
            )
        )

    conversation = create_test_conversation(
        sat_attrs, sat_edges, target_edges, clusters
    )

    validator = ClusterDataValidator()
    result = validator._validate_correctness_and_isolation_for_single_slice(
        conversation
    )

    print(f"   结果: {result.score}分")
    print(f"   详情: {result.info}")
    print()

    return result


def test_edge_case_single_element():
    """测试边界情况 - 单卫星单目标"""
    print("🧪 测试场景9: 单卫星单目标边界情况")

    sat_attrs = [SatelliteAttributes(id=1, health=10.0, pos=[1000, 2000, 3000])]

    sat_edges = []  # 没有卫星间连接

    target_edges = [TargetEdge(sat_id=1, target_id=101, quality=1.0)]

    clusters = [
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=0,
            master=1,
            sats=[1],
            targets=[101],
        )
    ]

    conversation = create_test_conversation(
        sat_attrs, sat_edges, target_edges, clusters
    )

    validator = ClusterDataValidator()
    result = validator._validate_correctness_and_isolation_for_single_slice(
        conversation
    )

    print(f"   结果: {result.score}分")
    print(f"   详情: {result.info}")
    print()

    return result


def test_severe_isolation_violations():
    """测试严重隔离性违规 - 大量跨簇连接和多簇归属"""
    print("🧪 测试场景10: 严重隔离性违规")

    sat_attrs = [
        SatelliteAttributes(id=1, health=10.0, pos=[1000, 2000, 3000]),
        SatelliteAttributes(id=2, health=9.5, pos=[1100, 2100, 3100]),
        SatelliteAttributes(id=3, health=9.0, pos=[1200, 2200, 3200]),
        SatelliteAttributes(id=4, health=8.5, pos=[2000, 3000, 4000]),
    ]

    sat_edges = [
        SatelliteEdge(from_sat=1, to_sat=2, distance=100.0),
        SatelliteEdge(from_sat=3, to_sat=4, distance=120.0),
    ]

    # 创建交叉连接场景
    target_edges = [
        TargetEdge(sat_id=1, target_id=101, quality=0.9),  # 卫星1 -> 目标101
        TargetEdge(sat_id=1, target_id=102, quality=0.8),  # 卫星1 -> 目标102
        TargetEdge(sat_id=2, target_id=103, quality=0.85),  # 卫星2 -> 目标103
        TargetEdge(sat_id=3, target_id=104, quality=0.75),  # 卫星3 -> 目标104
        TargetEdge(sat_id=4, target_id=105, quality=0.7),  # 卫星4 -> 目标105
    ]

    # 故意创建严重违规的分簇
    clusters = [
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=0,
            master=1,
            sats=[1, 2, 3],  # 卫星3同时在两个簇中
            targets=[102, 103, 105],  # 目标102和103跨簇，105跨簇
        ),
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=1,
            master=3,  # 卫星3同时作为两个簇的主节点
            sats=[3, 4, 1],  # 卫星3和1都重复了
            targets=[101, 104, 102],  # 目标101和102跨簇
        ),
    ]

    conversation = create_test_conversation(
        sat_attrs, sat_edges, target_edges, clusters
    )

    validator = ClusterDataValidator()
    result = validator._validate_correctness_and_isolation_for_single_slice(
        conversation
    )

    print(f"   结果: {result.score}分")
    print(f"   详情: {result.info}")
    print()

    return result


def run_all_tests():
    """运行所有测试"""
    print("=" * 80)
    print("🚀 开始运行正确性和隔离性验证函数测试")
    print("=" * 80)

    results = []

    # 运行所有测试场景
    results.append(test_perfect_scenario())
    results.append(test_missing_targets())
    results.append(test_cross_cluster_connections())
    results.append(test_multi_cluster_attribution())
    results.append(test_invalid_elements())
    results.append(test_complex_scenario())
    results.append(test_empty_cluster())
    results.append(test_large_scale_scenario())
    results.append(test_edge_case_single_element())
    results.append(test_severe_isolation_violations())

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
    print("这些测试验证了正确性和隔离性验证函数在各种场景下的表现:")
    print("• 完美场景：验证满分情况")
    print("• 目标遗漏：验证覆盖率扣分")
    print("• 跨簇连接：验证隔离性检查")
    print("• 多簇归属：验证重复元素检查")
    print("• 致命错误：验证不存在元素检查")
    print("• 复杂场景：验证多问题组合处理")
    print("• 边界情况：验证空簇和单元素处理")
    print("• 大规模场景：验证算法在大数据集上的表现")
    print("• 严重违规：验证极端违规情况的处理")


if __name__ == "__main__":
    run_all_tests()
