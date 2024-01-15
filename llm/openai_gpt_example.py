import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv


_ = load_dotenv(find_dotenv())

client = OpenAI(
    api_key=os.environ["OPENAI_SECRET_KEY"],
    base_url=os.environ["OPENAI_API_BASE"]
)

completion = client.chat.completions.create(
    # 创建一个 ChatCompletion
    # 调用模型：ChatGPT-3.5
    model="gpt-3.5-turbo",
    # message 是你的 prompt
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)
print(completion)