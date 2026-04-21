import json
import requests
import datetime



url = "https://openrouter.ai/api/v1/credits"
headers = {"Authorization": "Bearer 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"}
response = requests.get(url, headers=headers)
print(datetime.datetime.now(), json.dumps(response.json()))
