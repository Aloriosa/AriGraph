from openai import OpenAI
apikey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImFhZDEyY2YyLWIyZDAtNDdjOC1iMjg3LTIzNzkyMTBkYmIyOSIsImxvZ2dpbmdfaW5fdG9rZW4iOmZhbHNlLCJpYXQiOjE3NzI2MTI3MjksImV4cCI6MTc3MzIxNzUyOX0.vkbCVooDTW1P7XIMW6f9PECbenH0OulvyAa98wc3I9A'
client = OpenAI(base_url="https://inference.airi.net:46783/v1", api_key=apikey)
print(f"client {client}")
print(f"client.models.list {client.models}")
for i, model in enumerate(client.models.list(), start=1):
    print(f"\t{i})", model.id, "Max length:", model.max_model_len)