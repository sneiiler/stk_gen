# 卫星数据蒸馏系统

这是一个专门用于处理和优化卫星分簇训练数据的系统。该系统使用 OpenAI API 实现，能够将原始的卫星观测数据转换为结构化的训练数据。

## 功能特点

1. 数据处理能力：
   - 支持批量处理卫星观测数据
   - 自动处理时间序列数据
   - 支持多种分簇策略（balance/quality）
   - 实时保存处理结果

2. 智能分簇：
   - 基于卫星健康状态的高斯采样
   - 支持星间链路强度评估
   - 自动计算并优化目标观测质量
   - 灵活的分簇策略选择

3. 错误处理：
   - 实时错误记录
   - 自动重试机制
   - 数据验证和清理

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置

1. 创建 `.env` 文件并设置以下环境变量：

```bash
# OpenAI API配置
OPENAI_API_BASE=your-api-base-url-here
OPENAI_API_KEY=your-api-key-here

# 模型配置
OPENAI_MODEL_NAME=your-model-name  # 默认为 'o4-mini'
```

注意：
- `OPENAI_API_BASE` 是你的 OpenAI 兼容接口的基础 URL
- `OPENAI_API_KEY` 是你的 API 密钥
- 这两个配置项都是必需的

## 使用方法

1. 准备输入数据：
   - 确保数据符合指定的JSON格式
   - 包含卫星属性、星间链路和目标观测数据

2. 运行数据处理：

```python
from distill.distill import DataDistiller
from pathlib import Path

# 初始化蒸馏器
distiller = DataDistiller()

# 加载数据
input_file = Path("data/training_data_raw.json")
batch_data = load_json_data(input_file)

# 设置输出文件
output_file = Path("data/distilled_data.jsonl")

# 处理数据
distiller.process_batch(batch_data, output_file)
```

## 输入数据格式

```json
{
  "timestamp": "2025-06-06T04:01:10Z",
  "strategy": "balance",  // 或 "quality"
  "sat_attrs": [
    {
      "id": 143,
      "health": 0.85,  // 0.0-1.0
      "pos": [x, y, z]
    }
  ],
  "sat_edges": [
    {
      "from": 143,
      "to": 111,
      "w": 1.0  // 链路强度 0.0-1.0
    }
  ],
  "target_edges": [
    {
      "from": 143,
      "to": 1,
      "q": 1.0  // 观测质量 0.0-1.0
    }
  ]
}
```

## 输出格式

系统输出为JSONL格式，每行包含一个处理结果：

```json
{
  "chain_of_thought": "分析推理过程...",
  "result": [
    {
      "cluster_id": 1,
      "master": 143,
      "sats": [143, 111, 112],
      "targets": [1, 2]
    }
  ]
}
```

## 注意事项

1. 数据预处理：
   - 确保输入数据格式正确
   - 检查时间戳格式
   - 验证数值范围

2. 性能优化：
   - 合理设置批处理大小
   - 监控API调用频率
   - 定期备份处理结果

3. 错误处理：
   - 检查错误日志
   - 必要时调整重试策略
   - 保存未处理成功的数据

4. 最佳实践：
   - 定期更新API密钥
   - 监控模型性能
   - 备份原始数据 