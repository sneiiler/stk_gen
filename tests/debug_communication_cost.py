"""
调试通信代价计算

验证分簇后的通信代价是否正确计算
"""

import sys
from pathlib import Path

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


def debug_high_cost_scenario():
    """调试高通信代价场景"""
    print("🔍 调试测试场景4: 高通信代价")
    
    # 复制测试场景4的数据
    sat_attrs = [
        SatelliteAttributes(id=1, health=10.0, pos=[1000, 2000, 3000]),
        SatelliteAttributes(id=2, health=9.5, pos=[1100, 2100, 3100]),
        SatelliteAttributes(id=3, health=9.0, pos=[1200, 2200, 3200]),
        SatelliteAttributes(id=4, health=8.5, pos=[2000, 3000, 4000]),
        SatelliteAttributes(id=5, health=8.0, pos=[2100, 3100, 4100]),
        SatelliteAttributes(id=6, health=7.5, pos=[2200, 3200, 4200])
    ]
    
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
    
    clusters = [
        ClusterInfo(
            timestamp="2025-07-21T10:00:00Z",
            cluster_id=0,
            master=1,
            sats=[1, 2, 3, 4],  # 包含卫星4
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
    
    # 构建卫星距离字典
    sat_distances = {}
    all_satellites = {1, 2, 3, 4, 5, 6}
    
    for edge in sat_edges:
        sat_distances[(edge.from_sat, edge.to_sat)] = edge.distance
        sat_distances[(edge.to_sat, edge.from_sat)] = edge.distance  # 双向距离
    
    print("📋 网络连接信息：")
    for (sat1, sat2), distance in sorted(sat_distances.items()):
        if sat1 < sat2:  # 只显示一次
            print(f"   卫星{sat1} ↔ 卫星{sat2}: {distance}km")
    
    print("\n🏗️ 分簇信息：")
    for cluster in clusters:
        print(f"   簇{cluster.cluster_id}: 主节点{cluster.master}, 卫星{cluster.sats}")
    
    # 手动计算星座总通信代价
    print("\n💡 计算星座总通信代价（所有可联通的卫星对）：")
    validator = ClusterDataValidator()
    total_constellation_cost = 0
    constellation_connections = []
    
    for i, sat1 in enumerate(all_satellites):
        for sat2 in list(all_satellites)[i+1:]:
            path_cost = validator._find_shortest_path_cost(
                sat1, sat2, sat_distances, all_satellites
            )
            if path_cost is not None:
                total_constellation_cost += path_cost
                constellation_connections.append((sat1, sat2, path_cost))
                print(f"   卫星{sat1} → 卫星{sat2}: {path_cost}km")
    
    print(f"   星座总通信代价: {total_constellation_cost}km")
    
    # 手动计算分簇后的通信代价
    print("\n🔨 计算分簇后的通信代价：")
    
    # 簇内代价
    print("   簇内通信代价：")
    total_intra_cost = 0
    for cluster in clusters:
        cluster_sats = cluster.sats
        master_sat = cluster.master
        cluster_intra_cost = 0
        
        print(f"     簇{cluster.cluster_id}（主节点{master_sat}）：")
        for member_sat in cluster_sats:
            if member_sat == master_sat:
                continue
            
            path_cost = validator._find_shortest_path_cost(
                member_sat, master_sat, sat_distances, cluster_sats
            )
            if path_cost is not None:
                cluster_intra_cost += path_cost
                print(f"       卫星{member_sat} → 主节点{master_sat}: {path_cost}km")
        
        total_intra_cost += cluster_intra_cost
        print(f"     簇{cluster.cluster_id}内总代价: {cluster_intra_cost}km")
    
    # 主节点间代价
    print("   主节点间通信代价：")
    total_inter_cost = 0
    masters = [cluster.master for cluster in clusters]
    
    for i, master1 in enumerate(masters):
        for master2 in masters[i+1:]:
            path_cost = validator._find_shortest_path_cost(
                master1, master2, sat_distances, all_satellites
            )
            if path_cost is not None:
                total_inter_cost += path_cost
                print(f"       主节点{master1} → 主节点{master2}: {path_cost}km")
    
    total_cluster_cost = total_intra_cost + total_inter_cost
    
    print(f"\n📊 代价对比：")
    print(f"   分簇后总代价: {total_cluster_cost}km")
    print(f"   星座总代价: {total_constellation_cost}km")
    
    if total_constellation_cost > 0:
        ratio = total_cluster_cost / total_constellation_cost
        efficiency = (1 - ratio) * 100
        print(f"   代价比例: {ratio:.1%}")
        print(f"   效率提升: {efficiency:.1f}%")
        
        if ratio > 1.0:
            print("   ⚠️ 分簇后的代价比全联通网络还高！说明分簇策略有问题")
        else:
            print("   ✅ 分簇降低了通信代价")


if __name__ == "__main__":
    debug_high_cost_scenario()
