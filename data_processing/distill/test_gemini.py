from google import genai
from google.genai import types

http_options = types.HttpOptions(
    client_args={'proxy': 'socks5://127.0.0.1:1089'},
    async_client_args={'proxy': 'socks5://127.0.0.1:1089'},
)

client = genai.Client(api_key="AIzaSyBLO9aOoRS-16dNOIFjXGrG6SS7LceASlY", http_options=http_options)

response = client.models.generate_content(
    model="gemini-2.5-pro-preview-03-25", contents="你是谁"
)
print(response.text)