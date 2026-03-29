import os
from openai import OpenAI
from dotenv import load_dotenv # you'll need to pip install python-dotenv
from time import time
load_dotenv()

base_url = os.getenv("OPENAI_API_BASE_URL") # "https://inference.airi.net:46783/v1"
api_key = os.getenv("OPENAI_API_KEY") # "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImFhZDEyY2YyLWIyZDAtNDdjOC1iMjg3LTIzNzkyMTBkYmIyOSIsImxvZ2dpbmdfaW5fdG9rZW4iOmZhbHNlLCJpYXQiOjE3NzMwMDAwNjEsImV4cCI6MTc3MzYwNDg2MX0.xQYUhUzkgTjvFhUplVKuyu0j1T7IEVm-L90NDH1_LLE"
my_model = os.getenv("OPENAI_MODEL") # "Qwen/Qwen3-Next-80B-A3B-Instruct" #"Qwen/QwQ-32B"
client = OpenAI(base_url=base_url, api_key=api_key)
#print(f"client {client}")
#print(f"client.models.list {client.models}")
#for i, model in enumerate(client.models.list(), start=1):
#    print(f"\t{i})", model.id, "Max length:", model.max_model_len)

chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": 'hello',
                    }
                ],
                model=my_model,
            )
t0 = time()
response = chat_completion.choices[0].message.content
print(f"my model {my_model} Response: {chat_completion}")
print(f"Time taken: {time() - t0} seconds")