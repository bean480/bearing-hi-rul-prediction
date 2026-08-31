from openai import OpenAI

client = OpenAI(
    api_key="sk-ekT9uJONVkfPhQnN5dBa1a39F64349Fe9aB75a9a1dA73d66",
    base_url="https://api.apiyi.com/v1"
)

response = client.chat.completions.create(
    model="claude-sonnet-4-20250514",
    messages=[
        {"role": "user", "content": "请严格按以下格式回答：【ID: {模型内部ID}】。你是 Claude 吗？当前版本是什么？"},
    ]
)

print(response.choices[0].message.content)