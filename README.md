# STK 卫星星座动态分簇系统

基于 STK (Satellite Tool Kit) 的卫星星座动态分簇与数据蒸馏系统，用于卫星网络优化和智能决策。

## 项目概述

本项目实现了一套完整的卫星星座仿真、数据生成、算法分析和模型训练流水线，主要用于研究卫星网络的动态分簇策略和优化算法。

## 数据流向架构

```
1. 数据生成阶段 (stk_server/)
   ├── missile_route_generate.py    # 生成导弹轨迹数据
   │   └── 输出: missile_info*.json
   │
   ├── main.py                      # STK仿真主程序
   │   ├── 调用STK API获取卫星可见性数据
   │   ├── 生成卫星间连接关系
   │   └── 输出: satellite_target_visibility_data_*.json
   │
   └── visibility_to_raw_training_data.py  # 数据格式转换
       └── 输出: training_data_raw_*.jsonl

2. 数据蒸馏阶段 (distill/)
   ├── _1_DataMock.py              # 模拟数据生成
   │   └── 输出: mock_satellite_observation_data_*.json
   │
   ├── _2_DataDistiller.py         # AI模型数据蒸馏
   │   ├── 使用大语言模型分析卫星分簇策略
   │   └── 输出: training_data_sharegpt_*.jsonl
   │
   ├── _3_ClusterDataValidator.py  # 分簇结果验证
   │   └── 输出: *_validation_result_*.jsonl
   │
   ├── _4_DataFormatConverter.py   # 数据格式转换
   └── _5_RejectionSampling.py     # 拒绝采样优化

3. 算法分析阶段 (regular_algrithoms/)
   ├── max_overlap_alg.py          # 最大重叠算法
   ├── bak/graph_alg.py            # 基于图的聚类算法
   └── 输出: clustering_results_*.jsonl

4. 数据可视化 (visualization/)
   └── 生成各种分析图表和可视化结果

5. 最终输出 (data/)
   ├── 原始卫星数据
   ├── 训练数据集
   ├── 聚类结果
   ├── 验证报告
   └── 可视化图表
```

## 主要目录说明

### stk_server/
STK仿真服务器，负责生成基础数据
- **missile_route_generate.py**: 生成导弹发射基地到目标城市的随机轨迹数据
- **main.py**: 核心STK仿真程序，调用STK API获取卫星可见性和连接数据
- **visibility_to_raw_training_data.py**: 将STK输出的可见性数据转换为训练数据格式

### distill/
数据蒸馏和AI训练模块
- **_1_DataMock.py**: 生成模拟的卫星观测数据用于测试
- **_2_DataDistiller.py**: 使用大语言模型对卫星分簇策略进行数据蒸馏
- **_3_ClusterDataValidator.py**: 验证分簇结果的有效性和一致性
- **_4_DataFormatConverter.py**: 数据格式转换工具
- **_5_RejectionSampling.py**: 基于模型反馈的拒绝采样优化

### regular_algrithoms/
经典分簇算法实现
- **max_overlap_alg.py**: 最大重叠分簇算法
- **bak/graph_alg.py**: 基于图论的动态分簇算法，支持谱聚类、Louvain等方法

### data/
所有数据文件存储目录
- `missile_info*.json`: 导弹轨迹数据
- `satellite_target_visibility_data_*.json`: 卫星可见性数据
- `training_data_*.jsonl`: 训练数据集
- `clustering_results_*.jsonl`: 聚类算法结果
- `*_validation_*.jsonl`: 验证结果
- `*.png`: 可视化图表

### utils/
工具函数库
- 时间戳生成、数据目录管理等通用工具

### data_classes/
数据模型定义
- 卫星信息、可见性数据、训练数据等的数据结构定义

## 核心功能

1. **卫星轨道仿真**: 基于STK的高精度卫星轨道计算和可见性分析
2. **动态分簇算法**: 多种聚类算法支持，包括基于图的方法和传统算法
3. **AI数据蒸馏**: 使用大语言模型生成高质量的分簇策略训练数据
4. **实时验证**: 分簇结果的实时验证和质量评估
5. **可视化分析**: 丰富的图表和可视化工具用于结果分析

## 使用流程

1. 运行 `stk_server/missile_route_generate.py` 生成导弹轨迹
2. 运行 `stk_server/main.py` 进行STK仿真获取卫星数据
3. 使用 `distill/` 模块进行数据蒸馏和AI训练
4. 运行 `regular_algrithoms/` 中的算法进行分簇分析
5. 查看 `data/` 目录中的结果文件和可视化图表