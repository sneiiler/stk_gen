# 卫星分簇结果验证器使用说明

## 概述

`SatelliteClusterValidator` 是一个专门用于验证大模型生成的卫星分簇结果的验证器。它能够检查输出结果是否符合业务规则和约束条件，确保分簇结果的可靠性和有效性。

## 主要功能

### 1. 基础格式验证
- 检查输出是否包含必需的 `clusters` 字段
- 验证每个集群是否包含必需的字段：`cluster_id`、`master`、`sats`、`targets`
- 检查数据类型是否正确

### 2. 目标覆盖验证
- 确保所有输入目标都被正确覆盖
- 检查是否存在不存在的目标
- 计算目标覆盖率

### 3. 卫星分配验证
- 检查卫星是否被重复分配
- 验证是否存在不存在的卫星
- 计算卫星利用率

### 4. 主节点验证
- 确保主节点在卫星列表中
- 检查主节点是否被重复使用

### 5. 策略约束验证
- 根据策略类型（`balanced`/`quality`）验证卫星数量约束
- 检查空集群

### 6. 链路质量验证
- 计算集群内平均链路强度
- 识别链路质量较低的集群

### 7. 观测质量验证
- 计算集群的平均观测质量
- 识别观测质量较低的集群

### 8. 健康度验证
- 检查主节点健康度
- 计算集群整体健康度

## 使用方法

### 基本用法

```python
from data_processing.data_validator import SatelliteClusterValidator, ValidationResult

# 创建验证器实例
validator = SatelliteClusterValidator()

# 验证输出结果
result = validator.validate_output(output_data, input_data)

# 检查验证结果
if result.is_valid:
    print("验证通过")
else:
    print("验证失败")
    for error in result.errors:
        print(f"错误: {error}")
    
    for warning in result.warnings:
        print(f"警告: {warning}")
```

### 使用便捷函数

```python
from data_processing.data_validator import validate_satellite_clustering, generate_report

# 直接验证
result = validate_satellite_clustering(output_data, input_data)

# 生成验证报告
report = generate_report(result)
print(report)
```

### 带日志的验证

```python
import logging
from data_processing.data_validator import SatelliteClusterValidator

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建带日志的验证器
validator = SatelliteClusterValidator(logger)
result = validator.validate_output(output_data, input_data)
```

## 输入数据格式

### 输入数据 (input_data)
```json
{
    "timestamp": "2025-06-27T01:41:21Z",
    "strategy": "quality",
    "sat_attrs": [
        {
            "id": 132,
            "health": 0.87,
            "pos": [5231.847, -1289.387, -2519.124]
        }
    ],
    "sat_edges": [
        {
            "from": 125,
            "to": 166,
            "w": 0.94
        }
    ],
    "target_edges": [
        {
            "from": 116,
            "to": 5,
            "q": 0.86
        }
    ]
}
```

### 输出数据 (output_data)
```json
{
    "clusters": [
        {
            "cluster_id": 1,
            "master": 125,
            "sats": [125, 166, 165],
            "targets": [2]
        }
    ]
}
```

## 验证结果

### ValidationResult 结构
```python
@dataclass
class ValidationResult:
    is_valid: bool          # 验证是否通过
    errors: List[str]       # 错误信息列表
    warnings: List[str]     # 警告信息列表
    details: Dict[str, Any] # 详细信息
```

### 详细信息字段
- `target_coverage`: 目标覆盖信息
  - `input_targets`: 输入目标数量
  - `output_targets`: 输出目标数量
  - `coverage_rate`: 覆盖率

- `satellite_assignment`: 卫星分配信息
  - `total_satellites`: 总卫星数量
  - `assigned_satellites`: 已分配卫星数量
  - `utilization_rate`: 利用率

- `link_quality`: 链路质量信息
  - `overall_avg_strength`: 整体平均链路强度
  - `cluster_count`: 集群数量

- `observation_quality`: 观测质量信息
  - `overall_avg_quality`: 整体平均观测质量
  - `cluster_count`: 集群数量

## 验证规则

### 错误级别问题
1. **输出格式错误**: 缺少必需字段或数据类型错误
2. **目标覆盖不完整**: 缺少目标或包含不存在的目标
3. **卫星重复分配**: 同一卫星被分配到多个集群
4. **主节点错误**: 主节点不在卫星列表中或被重复使用
5. **空集群**: 集群没有卫星

### 警告级别问题
1. **卫星利用率低**: 有未使用的卫星
2. **策略约束违反**: 卫星数量超过策略限制
3. **链路质量低**: 平均链路强度低于0.3
4. **观测质量低**: 平均观测质量低于0.5
5. **健康度低**: 主节点健康度低于0.7或集群平均健康度低于0.6

## 示例报告

```
==================================================
卫星分簇结果验证报告
==================================================
验证状态: ✅ 通过

⚠️ 警告信息:
  - 集群 3 平均链路强度较低: 0.285
  - 集群 5 主节点 114 健康度较低: 0.650

📊 详细信息:
  target_coverage:
    input_targets: 25
    output_targets: 25
    coverage_rate: 1.0
  satellite_assignment:
    total_satellites: 25
    assigned_satellites: 20
    utilization_rate: 0.8
  link_quality:
    overall_avg_strength: 0.623
    cluster_count: 7
  observation_quality:
    overall_avg_quality: 0.712
    cluster_count: 7

==================================================
```

## 注意事项

1. **性能考虑**: 验证器会遍历所有数据，对于大规模数据集可能需要较长时间
2. **内存使用**: 验证过程中会创建多个映射表，注意内存使用
3. **错误处理**: 验证器会捕获异常并记录到日志中
4. **扩展性**: 可以通过继承 `SatelliteClusterValidator` 类来添加自定义验证规则

## 故障排除

### 常见问题

1. **导入错误**: 确保 `data_processing` 目录在 Python 路径中
2. **类型错误**: 检查输入数据的类型和格式
3. **内存不足**: 对于大数据集，考虑分批处理
4. **日志配置**: 确保日志记录器正确配置

### 调试建议

1. 使用详细的日志输出来跟踪验证过程
2. 检查验证结果的详细信息字段
3. 逐步验证各个验证步骤
4. 使用测试数据验证功能 