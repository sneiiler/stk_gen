"""
测试通信代价验证函数

该测试文件针对优化后的 `_validate_communication_cost_for_single_slice` 函数进行全面测试，
验证函数在各种通信代价场景下的表现。

使用方法:
    直接运行此文件即可执行所有测试:
    ```bash
    python test_communication_cost_validation.py
    ```

测试场景包括:
    1. 完美通信网络 - 验证满分情况
    2. 主节点无效 - 验证致命错误处理
    3. 孤星存在 - 验证连通性检查
    4. 高通信代价 - 验证代价扣分机制
    5. 复杂网络拓扑 - 验证路径计算
    6. 边界情况 - 验证异常处理
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


def test_perfect_communication_network():
    """测试完美通信网络 - 应该得到满分100分"""
    print("🧪 测试场景1: 完美通信网络 (期望得分: 100分)")
    
    # 创建6颗卫星，形成两个三角形连通网络
    sat_attrs = [
        SatelliteAttributes(id=1, health=10.0, pos=[1000, 2000, 3000]),
        SatelliteAttributes(id=2, health=9.5, pos=[1100, 2100, 3100]),
        SatelliteAttributes(id=3, health=9.0, pos=[1200, 2200, 3200]),
        SatelliteAttributes(id=4, health=8.5, pos=[2000, 3000, 4000]),
        SatelliteAttributes(id=5, health=8.0, pos=[2100, 3100, 4100]),
        SatelliteAttributes(id=6, health=7.5, pos=[2200, 3200, 4200])
    ]
    
    # 创建良好的连通网络（三角形拓扑，低代价）
    sat_edges = [
        # 第一个三角形 (1-2-3)
        SatelliteEdge(from_sat=1, to_sat=2, distance=50.0),
        SatelliteEdge(from_sat=2, to_sat=3, distance=55.0),
        SatelliteEdge(from_sat=3, to_sat=1, distance=60.0),
        # 第二个三角形 (4-5-6)
        SatelliteEdge(from_sat=4, to_sat=5, distance=45.0),
        SatelliteEdge(from_sat=5, to_sat=6, distance=50.0),
        SatelliteEdge(from_sat=6, to_sat=4, distance=55.0),
        # 跨簇连接（主节点间）
        SatelliteEdge(from_sat=1, to_sat=4, distance=200.0)
    ]
    
    # 创建目标连接
    target_edges = [
        TargetEdge(sat_id=1, target_id=101, quality=0.9),
        TargetEdge(sat_id=2, target_id=102, quality=0.8),
        TargetEdge(sat_id=3, target_id=103, quality=0.85),
        TargetEdge(sat_id=4, target_id=104, quality=0.75),
        TargetEdge(sat_id=5, target_id=105, quality=0.8),
        TargetEdge(sat_id=6, target_id=106, quality=0.9)
    ]
    
    # 创建合理的分簇（低通信代价）
    clusters = [
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=0,
            master=1,  # 主节点在簇内
            sats=[1, 2, 3],
            targets=[101, 102, 103]
        ),
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=1,
            master=4,  # 主节点在簇内
            sats=[4, 5, 6],
            targets=[104, 105, 106]
        )
    ]
    
    conversation = create_test_conversation(sat_attrs, sat_edges, target_edges, clusters)
    
    validator = ClusterDataValidator()
    result = validator._validate_communication_cost_for_single_slice(conversation)
    
    print(f"   结果: {result.score}分")
    print(f"   详情: {result.info}")
    print()
    
    return result


def test_invalid_master_node():
    """测试主节点无效场景 - 主节点不在簇内卫星列表中"""
    print("🧪 测试场景2: 主节点无效 (期望得分: 0分)")
    
    sat_attrs = [
        SatelliteAttributes(id=1, health=10.0, pos=[1000, 2000, 3000]),
        SatelliteAttributes(id=2, health=9.5, pos=[1100, 2100, 3100]),
        SatelliteAttributes(id=3, health=9.0, pos=[1200, 2200, 3200])
    ]
    
    sat_edges = [
        SatelliteEdge(from_sat=1, to_sat=2, distance=100.0),
        SatelliteEdge(from_sat=2, to_sat=3, distance=120.0)
    ]
    
    target_edges = [
        TargetEdge(sat_id=1, target_id=101, quality=0.9),
        TargetEdge(sat_id=2, target_id=102, quality=0.8),
        TargetEdge(sat_id=3, target_id=103, quality=0.85)
    ]
    
    # 主节点4不在簇内卫星列表中
    clusters = [
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=0,
            master=4,  # 主节点4不在下面的sats列表中
            sats=[1, 2, 3],  # 没有卫星4
            targets=[101, 102, 103]
        )
    ]
    
    conversation = create_test_conversation(sat_attrs, sat_edges, target_edges, clusters)
    
    validator = ClusterDataValidator()
    result = validator._validate_communication_cost_for_single_slice(conversation)
    
    print(f"   结果: {result.score}分")
    print(f"   详情: {result.info}")
    print()
    
    return result


def test_isolated_satellites():
    """测试孤星存在场景 - 簇内卫星无法连通主节点"""
    print("🧪 测试场景3: 孤星存在 (期望得分: 0分)")
    
    sat_attrs = [
        SatelliteAttributes(id=1, health=10.0, pos=[1000, 2000, 3000]),
        SatelliteAttributes(id=2, health=9.5, pos=[1100, 2100, 3100]),
        SatelliteAttributes(id=3, health=9.0, pos=[1200, 2200, 3200]),
        SatelliteAttributes(id=4, health=8.5, pos=[2000, 3000, 4000])
    ]
    
    # 创建不完全连通的网络
    sat_edges = [
        SatelliteEdge(from_sat=1, to_sat=2, distance=100.0),
        # 卫星3和4与其他卫星没有连接，形成孤星
    ]
    
    target_edges = [
        TargetEdge(sat_id=1, target_id=101, quality=0.9),
        TargetEdge(sat_id=2, target_id=102, quality=0.8),
        TargetEdge(sat_id=3, target_id=103, quality=0.85),
        TargetEdge(sat_id=4, target_id=104, quality=0.75)
    ]
    
    # 卫星3和4无法连通主节点1
    clusters = [
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=0,
            master=1,
            sats=[1, 2, 3, 4],  # 卫星3和4无法连通主节点1
            targets=[101, 102, 103, 104]
        )
    ]
    
    conversation = create_test_conversation(sat_attrs, sat_edges, target_edges, clusters)
    
    validator = ClusterDataValidator()
    result = validator._validate_communication_cost_for_single_slice(conversation)
    
    print(f"   结果: {result.score}分")
    print(f"   详情: {result.info}")
    print()
    
    return result


def test_high_communication_cost():
    """测试高通信代价场景 - 代价占星座总代价比例过高"""
    print("🧪 测试场景4: 高通信代价 (期望扣分)")
    
    sat_attrs = [
        SatelliteAttributes(id=1, health=10.0, pos=[1000, 2000, 3000]),
        SatelliteAttributes(id=2, health=9.5, pos=[1100, 2100, 3100]),
        SatelliteAttributes(id=3, health=9.0, pos=[1200, 2200, 3200]),
        SatelliteAttributes(id=4, health=8.5, pos=[2000, 3000, 4000]),
        SatelliteAttributes(id=5, health=8.0, pos=[2100, 3100, 4100]),
        SatelliteAttributes(id=6, health=7.5, pos=[2200, 3200, 4200])
    ]
    
    # 创建高代价网络（距离很大）
    sat_edges = [
        # 簇内连接（相对较近）
        SatelliteEdge(from_sat=1, to_sat=2, distance=50.0),
        SatelliteEdge(from_sat=2, to_sat=3, distance=60.0),
        SatelliteEdge(from_sat=4, to_sat=5, distance=55.0),
        SatelliteEdge(from_sat=5, to_sat=6, distance=65.0),
        # 全局连接（很远，创建高代价）
        SatelliteEdge(from_sat=1, to_sat=4, distance=5000.0),  # 非常高的代价
        SatelliteEdge(from_sat=2, to_sat=5, distance=100.0),
        SatelliteEdge(from_sat=3, to_sat=6, distance=110.0)
    ]
    
    target_edges = [
        TargetEdge(sat_id=1, target_id=101, quality=0.9),
        TargetEdge(sat_id=2, target_id=102, quality=0.8),
        TargetEdge(sat_id=3, target_id=103, quality=0.85),
        TargetEdge(sat_id=4, target_id=104, quality=0.75),
        TargetEdge(sat_id=5, target_id=105, quality=0.8),
        TargetEdge(sat_id=6, target_id=106, quality=0.9)
    ]
    
    # 分簇会导致高通信代价（主节点间距离很远）
    clusters = [
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=0,
            master=1,
            sats=[1, 2, 3,4],
            targets=[101, 102, 103]
        ),
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=1,
            master=5,
            sats=[5, 6],
            targets=[104, 105, 106]
        )
    ]
    
    conversation = create_test_conversation(sat_attrs, sat_edges, target_edges, clusters)
    
    validator = ClusterDataValidator()
    result = validator._validate_communication_cost_for_single_slice(conversation)
    
    print(f"   结果: {result.score}分")
    print(f"   详情: {result.info}")
    print()
    
    return result


def test_complex_network_topology():
    """测试复杂网络拓扑 - 多跳路径和路径选择"""
    print("🧪 测试场景5: 复杂网络拓扑")
    
    # 创建5颗卫星的链状网络
    sat_attrs = [
        SatelliteAttributes(id=1, health=10.0, pos=[1000, 2000, 3000]),
        SatelliteAttributes(id=2, health=9.5, pos=[1100, 2100, 3100]),
        SatelliteAttributes(id=3, health=9.0, pos=[1200, 2200, 3200]),
        SatelliteAttributes(id=4, health=8.5, pos=[1300, 2300, 3300]),
        SatelliteAttributes(id=5, health=8.0, pos=[1400, 2400, 3400])
    ]
    
    # 创建链状连接，需要多跳路径
    sat_edges = [
        SatelliteEdge(from_sat=1, to_sat=2, distance=80.0),
        SatelliteEdge(from_sat=2, to_sat=3, distance=85.0),
        SatelliteEdge(from_sat=3, to_sat=4, distance=90.0),
        SatelliteEdge(from_sat=4, to_sat=5, distance=95.0),
        # 添加一个捷径连接
        SatelliteEdge(from_sat=1, to_sat=4, distance=200.0)  # 直连但代价较高
    ]
    
    target_edges = [
        TargetEdge(sat_id=1, target_id=101, quality=0.9),
        TargetEdge(sat_id=2, target_id=102, quality=0.8),
        TargetEdge(sat_id=3, target_id=103, quality=0.85),
        TargetEdge(sat_id=4, target_id=104, quality=0.75),
        TargetEdge(sat_id=5, target_id=105, quality=0.8)
    ]
    
    # 主节点1需要通过多跳路径到达卫星5
    clusters = [
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=0,
            master=1,
            sats=[1, 2, 3, 4, 5],  # 卫星5需要通过多跳路径连接
            targets=[101, 102, 103, 104, 105]
        )
    ]
    
    conversation = create_test_conversation(sat_attrs, sat_edges, target_edges, clusters)
    
    validator = ClusterDataValidator()
    result = validator._validate_communication_cost_for_single_slice(conversation)
    
    print(f"   结果: {result.score}分")
    print(f"   详情: {result.info}")
    print()
    
    return result


def test_edge_case_single_cluster():
    """测试边界情况 - 单个卫星单个簇"""
    print("🧪 测试场景6: 单卫星单簇边界情况")
    
    sat_attrs = [
        SatelliteAttributes(id=1, health=10.0, pos=[1000, 2000, 3000])
    ]
    
    sat_edges = []  # 没有卫星间连接
    
    target_edges = [
        TargetEdge(sat_id=1, target_id=101, quality=1.0)
    ]
    
    clusters = [
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=0,
            master=1,
            sats=[1],
            targets=[101]
        )
    ]
    
    conversation = create_test_conversation(sat_attrs, sat_edges, target_edges, clusters)
    
    validator = ClusterDataValidator()
    result = validator._validate_communication_cost_for_single_slice(conversation)
    
    print(f"   结果: {result.score}分")
    print(f"   详情: {result.info}")
    print()
    
    return result


def test_multiple_issues():
    """测试多重问题场景 - 同时包含多种通信问题"""
    print("🧪 测试场景7: 多重问题场景")
    
    sat_attrs = [
        SatelliteAttributes(id=1, health=10.0, pos=[1000, 2000, 3000]),
        SatelliteAttributes(id=2, health=9.5, pos=[1100, 2100, 3100]),
        SatelliteAttributes(id=3, health=9.0, pos=[1200, 2200, 3200]),
        SatelliteAttributes(id=4, health=8.5, pos=[2000, 3000, 4000]),
        SatelliteAttributes(id=5, health=8.0, pos=[2100, 3100, 4100])
    ]
    
    # 部分连通的网络
    sat_edges = [
        SatelliteEdge(from_sat=1, to_sat=2, distance=100.0),
        # 卫星3孤立，卫星4和5连通但与1,2不连通
        SatelliteEdge(from_sat=4, to_sat=5, distance=150.0)
    ]
    
    target_edges = [
        TargetEdge(sat_id=1, target_id=101, quality=0.9),
        TargetEdge(sat_id=2, target_id=102, quality=0.8),
        TargetEdge(sat_id=3, target_id=103, quality=0.85),
        TargetEdge(sat_id=4, target_id=104, quality=0.75),
        TargetEdge(sat_id=5, target_id=105, quality=0.8)
    ]
    
    # 问题1: 主节点6不存在，问题2: 卫星3孤立，问题3: 卫星4,5无法连通主节点1
    clusters = [
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=0,
            master=6,  # 主节点6不存在
            sats=[1, 2, 3],  # 卫星3孤立
            targets=[101, 102, 103]
        ),
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=1,
            master=1,  # 主节点1不在簇内
            sats=[4, 5],  # 主节点不在簇内
            targets=[104, 105]
        )
    ]
    
    conversation = create_test_conversation(sat_attrs, sat_edges, target_edges, clusters)
    
    validator = ClusterDataValidator()
    result = validator._validate_communication_cost_for_single_slice(conversation)
    
    print(f"   结果: {result.score}分")
    print(f"   详情: {result.info}")
    print()
    
    return result


def run_all_tests():
    """运行所有测试"""
    print("=" * 80)
    print("🚀 开始运行通信代价验证函数测试")
    print("=" * 80)
    
    results = []
    
    # 运行所有测试场景
    results.append(test_perfect_communication_network())
    results.append(test_invalid_master_node())
    results.append(test_isolated_satellites())
    results.append(test_high_communication_cost())
    results.append(test_complex_network_topology())
    results.append(test_edge_case_single_cluster())
    results.append(test_multiple_issues())
    
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
    print("这些测试验证了通信代价验证函数在各种场景下的表现:")
    print("• 完美网络：验证低代价高分情况")
    print("• 主节点无效：验证致命错误检测")
    print("• 孤星存在：验证连通性检查")
    print("• 高通信代价：验证代价扣分机制")
    print("• 复杂拓扑：验证多跳路径计算")
    print("• 边界情况：验证最小规模处理")
    print("• 多重问题：验证综合错误处理")


if __name__ == "__main__":
    run_all_tests()
