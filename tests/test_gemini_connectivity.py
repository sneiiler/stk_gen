# from google import genai
# from google.genai import types

# http_options = types.HttpOptions(
#     client_args={'proxy': 'socks5://127.0.0.1:1089'},
#     async_client_args={'proxy': 'socks5://127.0.0.1:1089'},
# )

# client = genai.Client(api_key="AIzaSyBLO9aOoRS-16dNOIFjXGrG6SS7LceASlY", http_options=http_options)

# response = client.models.generate_content(
#     model="gemini-2.5-pro", contents="你是谁"
# )
# print(response.text)

import openai
import httpx
import sys
from icecream import ic
# 设置代理配置
proxy_config = httpx.Proxy("socks5://127.0.0.1:1089")

GEMINI_API_KEY="AIzaSyAc8m9lfsDKgZZofjjIPmdUTusJOqDJRck"
GEMINI_API_BASE="https://generativelanguage.googleapis.com/v1beta/openai/"
# 创建 OpenAI 客户端，配置代理
client = openai.OpenAI(
    api_key=GEMINI_API_KEY,
    http_client=httpx.Client(proxy=proxy_config),
    base_url=GEMINI_API_BASE
)

# 发送流式请求
response = client.chat.completions.create(
    model="gemini-2.5-pro",  # 或者使用 "gpt-4" 等其他模型
    messages=[
        {"role": "user", "content": "你是谁"}
    ],
    stream=True,  # 启用流式传输
    extra_body={
      'extra_body': {
        "google": {
          "thinking_config": {
            "thinking_budget": 800,
            "include_thoughts": True
          }
        }
      }
    }
)

# 流式处理响应
print("AI 回复：", end="", flush=True)
for chunk in response:
    ic(chunk.choices[0].delta)
    if chunk.choices[0].delta.content is not None:
        content = chunk.choices[0].delta.content
        print(content, end="", flush=True)
        
print()  # 换行