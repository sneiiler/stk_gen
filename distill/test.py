import os
from openai import OpenAI

API_SECRET_KEY = "sk-zk2030368d85cdb45dea4d38a02f8452859685148b72abc0"
BASE_URL = "https://api.zhizengzeng.com/v1/chat/completions"
# 智增增的base-url

os.environ["OPENAI_API_KEY"] = API_SECRET_KEY
os.environ["OPENAI_API_BASE"] = BASE_URL
client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)

completion = client.chat.completions.create(
  model="o4-mini",
  messages=[
    {"role": "system", "content": "你是一个专门用于卫星分簇的AI助手。你需要根据输入的卫星数据，按照要求进行分簇并输出结果。"},
    {"role": "user", "content": """我们正在进行大规模星座动态分簇，目标是根据卫星间的链路强度(即数据中的sat_edges.w)、目标的覆盖情况以及卫星的健康状态 
来优化分簇。
任务的目标是将N颗卫星划分为若干个簇，每个簇包含一个主星（簇头，可以主要考虑卫星的健康状况）和若干成员星，
确保每个目标至少被簇内的一颗卫星覆盖，并且簇内的星间时延总和最小。

关键要求：
1. Chain-of-Thought推理：解释每个簇决策的关键因素。比如，为什么选择某些卫星作为簇头，如何考虑链路强度、目标覆盖和卫 
星健康状态来优化簇的组成。
2. 考虑策略：
   - 当策略为"balanced"时，只要保证至少每个目标至少有一个卫星观测即可，并不是所有的卫星都需要执行观测任务
   - 当策略为"quality"时，将尽可能多的卫星划分到星簇内，当确定好要执行观测任务的卫星之后，如果其他卫星与当前执行卫星
的连通度较好并且没有被占用，这意味着他们距离比较近，可能在下一个时间段就能观测到目标，这个时候就可以提前将这些状态好
的卫星纳入到星簇
3. 考虑卫星健康状态：在分簇决策时，卫星的健康状态应作为一个重要因素。健康状态较低的卫星可能需要被排除在簇头之外。   
4. 输出清晰：确保每个簇的结构清晰，包括簇ID、簇头卫星、簇内卫星、覆盖的目标等信息。

输入数据：
{'timestamp': '2025-06-06T04:01:10Z', 'strategy': 'balanced', 'sat_attrs': [{'id': 143, 'health': 10, 'pos': [6044.846, 3931.97, 3172.458]}], 'sat_edges': [{'from': 143, 'to': 111, 'w': 1.0}, {'from': 143, 'to': 112, 'w': 0.5}, {'from': 143, 'to': 133, 'w': 0.44}, {'from': 143, 'to': 134, 'w': 0.43}, {'from': 143, 'to': 142, 'w': 0.42}, {'from': 
143, 'to': 144, 'w': 0.42}, {'from': 143, 'to': 152, 'w': 0.63}, {'from': 143, 'to': 153, 'w': 0.41}, {'from': 143, 
'to': 161, 'w': 0.89}, {'from': 143, 'to': 162, 'w': 0.37}, {'from': 143, 'to': 166, 'w': 0.43}], 'target_edges': [{'from': 143, 'to': 1, 'w': 1.0}]}

请按照以下格式输出：
<|chain_of_thought|>
这里是思考过程...
<|result|>
[
    {
    "cluster": id,
    "master": master_sat_id,
    "sats": [sat_id_1, sat_id_2, ...],
    "targets": [target_id_1, target_id_2, ...]
    }
]"""}
  ]
)

print(completion.choices[0].message)