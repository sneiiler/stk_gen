import random
import json
from datetime import datetime, timezone, timedelta
from utils.misc_utils import get_data_dir, get_current_timestamp  # 导入获取data目录和时间戳的工具函数

# 卫星编号：111-116, 121-126, ..., 161-166 共36颗
sat_ids = [111 + 10*r + c for r in range(0, 6) for c in range(6)]
# 目标编号：1-50 共50个
target_ids = list(range(1, 51))

def generate_dataset(timestamp):
    """
    Generate a single dataset for satellite-target observation simulation.

    Args:
        timestamp (str): 时间戳字符串。
    Returns:
        dict: A dataset containing timestamp, strategy, sat_attrs, sat_edges, and target_edges.
    """
    # 随机选择策略
    strategy = random.choice(["balance", "quailty"])
    
    # 生成卫星-卫星可见性边
    sat_edges = []
    num_sat_edges = random.randint(10, 30)  # 随机边数
    connected_sats = set()  # 记录出现过的卫星
    for _ in range(num_sat_edges):
        a, b = random.sample(sat_ids, 2)  # 随机选两颗不同卫星
        w = round(random.uniform(0.2, 1.0), 2)  # 通信权重w
        sat_edges.append({"from": a, "to": b, "w": w})
        connected_sats.update([a, b])  # 记录已连接卫星
    
    # 生成卫星属性，只为出现过的卫星生成
    sat_attrs = []
    for sat in connected_sats:
        health = round(random.uniform(0.5, 1.0), 2)  # 卫星健康度
        pos = [round(random.uniform(-8000, 8000), 3) for _ in range(3)]  # 卫星三维位置
        sat_attrs.append({"id": sat, "health": health, "pos": pos})
    
    # 生成卫星-目标观测边
    target_edges = []
    num_target_edges = random.randint(10, 40)  # 随机观测边数
    for _ in range(num_target_edges):
        sat = random.choice(list(connected_sats))  # 只用已连接卫星
        tgt = random.choice(target_ids)  # 随机目标
        q = round(random.uniform(0.2, 1.0), 2)  # 观测质量q
        target_edges.append({"from": sat, "to": tgt, "q": q})
    
    return {
        "timestamp": timestamp,
        "strategy": strategy,
        "sat_attrs": sat_attrs,
        "sat_edges": sat_edges,
        "target_edges": target_edges
    }

# 生成200组数据
# 每组数据结构为dict，包含时间戳、策略、卫星属性、卫星-卫星边、卫星-目标边
# 最终输出为JSON格式

datasets = []

now = datetime.now(timezone.utc)
for i in range(200):
    ts = (now + timedelta(seconds=10*i)).strftime("%Y-%m-%dT%H:%M:%SZ")
    datasets.append(generate_dataset(ts))

# 写入到data目录下的mock_satellite_observation_data_时间戳.json文件
file_ts = get_current_timestamp()
output_path = get_data_dir() / f"mock_satellite_observation_data_{file_ts}.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(datasets, f, ensure_ascii=False, indent=2)

# 打印输出所有数据
print(json.dumps(datasets, indent=2))
