🚀 卫星分簇优化蒸馏系统 - 快速使用指南

## 一分钟快速开始

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **配置API**
   编辑 `optimization_distiller_standalone.py` 第634行左右：
   ```python
   API_CONFIG = {
       "api_base": "https://your-api-base-url/v1/",  # 改成你的API地址
       "api_key": "your-api-key-here",              # 改成你的API密钥
       "proxy": None,                               # 如需代理: "socks5://127.0.0.1:1089"
   }
   ```

3. **准备数据**
   将验证数据文件放到 `data/validation_result.jsonl`

4. **运行**
   ```bash
   python optimization_distiller_standalone.py
   ```

## 主要参数调整

在main()函数中可调整：
- `model_name`: 模型名称 (默认: gemini-2.5-pro)
- `temperature`: 生成温度 (默认: 0.3)
- `max_workers`: 并发线程数 (默认: 4)
- `requests_per_minute`: 请求频率 (默认: 100)

## 输出文件

会生成两个文件：
- 完整数据：`optimization_distilled_sharegpt_*.jsonl`
- 训练格式：`sharegpt_optimization_distilled_sharegpt_*.jsonl`

## 支持的模型

- Gemini系列: gemini-2.5-pro, gemini-1.5-pro等
- OpenAI系列: gpt-4o, gpt-4-turbo等
- 其他兼容OpenAI API的模型

就这么简单！🎉
